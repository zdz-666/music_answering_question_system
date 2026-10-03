"""会话持久化：用 SQLite 文件替代原先 main.py 里的进程内字典。

原来 `sessions = {}` 有两个问题：
- `uvicorn --workers 2` 时每个 worker 各存一份，同一会话被路由到另一个 worker 就读不到历史；
- 进程重启即丢。

改用本地 SQLite 文件：跨进程共享、重启不丢，且不需要额外起服务（与嵌入式 Chroma 一致）。
多进程并发写靠 WAL + busy_timeout 兜住：WAL 让读写可以并存，busy_timeout 让写冲突时等待
而不是直接抛 database is locked。

这些函数保持同步实现：单次操作就是一次本地文件读写，复用连接后在亚毫秒级，与会话链路里动辄数秒的
LLM 调用不是一个量级，没必要再下沉线程池。

这里是三层记忆里的第一层（工作记忆）的落点：
- messages 表保存当前会话的最近消息，注入 prompt 的是最近 HISTORY_WINDOW 条；
- session_summaries 表保存一段滚动摘要，消息条数超过 COMPRESS_TRIGGER 时由
  memory/working.py 把最旧一批消息合并进摘要并删除，压缩走「claim → LLM → commit」
  两阶段，compression_claims 表保证多 worker 下同一会话不会被重复压缩。
"""

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

from config import SESSION_DB_PATH
from models import ChatMessage
from observability import logger

HISTORY_WINDOW = 4  # 生成答案时注入的历史消息条数（与原实现一致）
COMPRESS_BATCH = 10  # 每轮压缩最少处理的增量消息数
COMPRESS_TRIGGER = HISTORY_WINDOW + COMPRESS_BATCH  # 超过这个条数就触发摘要压缩
# 消息条数的兜底硬上限：正常路径永远不该触发（压缩会先把消息降回 HISTORY_WINDOW），
# 一旦触发说明压缩流程失效，这里保证消息不会无界增长，同时打 warning 暴露问题。
MAX_MESSAGES_HARD = 200
CLAIM_STALE_SECONDS = 60.0  # 压缩认领的过期秒数，兜住压缩过程中进程崩溃
BUSY_TIMEOUT = 5.0  # 写锁被其它 worker 占用时的最长等待秒数

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    timestamp  TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 压缩认领标记：同一会话同一时刻只允许一个 worker 在压缩
CREATE TABLE IF NOT EXISTS compression_claims (
    session_id TEXT PRIMARY KEY,
    claimed_at REAL NOT NULL
);
"""


_local = threading.local()


def _conn() -> sqlite3.Connection:
    """按线程复用一个长连接。

    两点原因：
    - sqlite3 的连接不能跨线程共享，而 FastAPI 会在多个工作线程里调用这里，所以按线程缓存；
    - 反复开关连接时，最后一个连接关闭会触发 WAL checkpoint 并删掉 -wal/-shm 文件，
      Windows 上这一下实测要 35ms 左右，成为单次操作的主要开销。连接复用后写入约 1.4ms、
      查询约 0.03ms。

    WAL 下连接长开不会阻塞其它进程写入，SQLite 也会按 wal_autocheckpoint 自动回收 -wal，
    不需要手工干预。
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(SESSION_DB_PATH, timeout=BUSY_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


@contextmanager
def _connect():
    """借出当前线程的连接：正常结束提交，出错回滚（两者都不关闭连接）。"""
    conn = _conn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    conn.commit()


def init_db() -> None:
    """建表并切到 WAL。幂等，多个 worker 同时启动也安全。"""
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)


def new_session_id() -> str:
    """时间戳 + 随机后缀。

    原先用 len(sessions) 当后缀，多 worker 下各进程各自计数，同一秒内会算出相同的 id；
    换成随机后缀后不需要任何共享计数器。
    """
    return f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


def get_or_create_session(session_id: str | None) -> str:
    """session_id 已存在就用它，否则新建一个并返回。"""
    with _connect() as conn:
        if session_id:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                return session_id

        session_id = new_session_id()
        conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, datetime.now().isoformat()),
        )
    return session_id


def add_message(session_id: str, message: ChatMessage) -> None:
    """追加一条消息，并只在超过兜底硬上限时裁剪。

    正常的瘦身交给 memory/working.py 的摘要压缩（把最旧一批消息并入摘要后删除），
    这里只是最后一道防线：真触发了说明压缩没跑起来，打 warning 暴露。
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, message.role, message.content, datetime.now().isoformat()),
        )
        cursor = conn.execute(
            """
            DELETE FROM messages
             WHERE session_id = ?
               AND id NOT IN (
                   SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
               )
            """,
            (session_id, session_id, MAX_MESSAGES_HARD),
        )
        trimmed = cursor.rowcount

    if trimmed:
        logger.warning(
            "messages.hard_trim",
            extra={"fields": {"session_id": session_id, "trimmed": trimmed}},
        )


def count_messages(session_id: str) -> int:
    """会话当前的消息条数，用于判断是否该做摘要压缩。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(row["total"])


def get_summary(session_id: str) -> str | None:
    """当前会话的滚动摘要；没有或为空时返回 None。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT summary FROM session_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    summary = (row["summary"] or "").strip()
    return summary or None


def list_oldest_messages(session_id: str, limit: int) -> list[dict]:
    """最早 limit 条消息，返回 [{"id", "role", "content"}]。

    带上 id 是为了压缩提交时能按 id 精确删除这一批（并发新插入的消息 id 更大，不受影响）。
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content FROM messages
             WHERE session_id = ?
             ORDER BY id ASC
             LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [
        {"id": row["id"], "role": row["role"], "content": row["content"]}
        for row in rows
    ]


def claim_compression(session_id: str) -> bool:
    """压缩两阶段的第一步：抢占该会话的压缩权。

    整件事只用一条 UPSERT 完成，靠 rowcount 判断结果：插入成功（此前没认领）或被刷新的
    过期认领都算抢占成功。DO UPDATE 上的 WHERE 让"未过期的认领"变成一次空操作
    （changed=0），于是并发调用里只有一个能拿到 True。

    不能写成"先 SELECT 再 INSERT"：WAL 下读快照与写入之间没有互斥，两个 worker
    同时读到"没有认领"就会双双抢占成功（实测如此）。单条 UPSERT 由 SQLite 串行化写入，
    第二个调用一定看到第一个的结果。
    """
    now = time.time()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO compression_claims (session_id, claimed_at) VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET claimed_at = excluded.claimed_at
             WHERE compression_claims.claimed_at < ?
            """,
            (session_id, now, now - CLAIM_STALE_SECONDS),
        )
        # rowcount：插入 1 / 刷新过期认领 1 / 命中未过期认领的 WHERE 而空操作 0
        return cursor.rowcount == 1


def release_compression(session_id: str) -> None:
    """压缩失败时释放认领：消息保留，下一轮再试。"""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM compression_claims WHERE session_id = ?", (session_id,)
        )


def commit_compression(session_id: str, summary: str, max_id: int) -> None:
    """压缩两阶段的第二步：写回摘要 + 删除已被摘要覆盖的那批消息 + 释放认领。

    三件事在同一个事务里完成，避免出现"摘要更新了但消息没删"或"消息删了但摘要没写"的中间态。
    删除用 `id <= max_id`：max_id 是本次读到的批次里最大的 id，等价于删掉这一批，
    压缩期间新插入的消息 id 更大，不会被误删。
    """
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO session_summaries (session_id, summary, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE
                SET summary = excluded.summary, updated_at = excluded.updated_at
            """,
            (session_id, summary, now),
        )
        conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id <= ?",
            (session_id, max_id),
        )
        conn.execute(
            "DELETE FROM compression_claims WHERE session_id = ?", (session_id,)
        )


def get_history_str(session_id: str) -> str:
    """最近 HISTORY_WINDOW 条消息拼成提示词里的历史段落，前面带上滚动摘要。

    摘要代表已被裁掉的那些轮次，最近几条代表刚刚发生的事，两者拼起来才是完整的工作记忆。
    """
    summary = get_summary(session_id)

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
             WHERE session_id = ?
             ORDER BY id DESC
             LIMIT ?
            """,
            (session_id, HISTORY_WINDOW),
        ).fetchall()

    history = f"【此前对话摘要】{summary}\n" if summary else ""
    for row in reversed(rows):  # 倒序取出，翻回时间正序
        if row["role"] == "user":
            history += f"用户: {row['content']}\n"
        elif row["role"] == "assistant":
            history += f"助手: {row['content']}\n"
    return history


def list_messages(session_id: str, number: int) -> list[ChatMessage] | None:
    """最近 number 条消息；会话不存在时返回 None，由调用方转 404。"""
    with _connect() as conn:
        if not conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone():
            return None

        rows = conn.execute(
            """
            SELECT role, content, timestamp FROM messages
             WHERE session_id = ?
             ORDER BY id DESC
             LIMIT ?
            """,
            (session_id, number),
        ).fetchall()

    return [
        ChatMessage(role=row["role"], content=row["content"], timestamp=row["timestamp"])
        for row in reversed(rows)
    ]


def clear_messages(session_id: str) -> bool:
    """清空消息与摘要；会话不存在返回 False。

    保留会话行是有意的：原实现里 delete 之后会话仍然存在，再查历史会返回空列表而不是 404。
    摘要必须一起清掉，否则"清空历史"之后助手仍然记得之前聊过什么。
    """
    with _connect() as conn:
        if not conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone():
            return False

        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute(
            "DELETE FROM session_summaries WHERE session_id = ?", (session_id,)
        )
        conn.execute(
            "DELETE FROM compression_claims WHERE session_id = ?", (session_id,)
        )
    return True