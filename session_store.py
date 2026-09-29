"""会话持久化：用 SQLite 文件替代原先 main.py 里的进程内字典。

原来 `sessions = {}` 有两个问题：
- `uvicorn --workers 2` 时每个 worker 各存一份，同一会话被路由到另一个 worker 就读不到历史；
- 进程重启即丢。

改用本地 SQLite 文件：跨进程共享、重启不丢，且不需要额外起服务（与嵌入式 Chroma 一致）。
多进程并发写靠 WAL + busy_timeout 兜住：WAL 让读写可以并存，busy_timeout 让写冲突时等待
而不是直接抛 database is locked。

这些函数保持同步实现：单次操作就是一次本地文件读写，复用连接后在亚毫秒级，与会话链路里动辄数秒的
LLM 调用不是一个量级，没必要再下沉线程池。
"""

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime

from config import SESSION_DB_PATH
from models import ChatMessage

MAX_MESSAGES = 20  # 单个会话最多保留的消息条数（与原实现的截断阈值一致）
HISTORY_WINDOW = 4  # 生成答案时注入的历史消息条数（与原实现一致）
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
    """追加一条消息，并把该会话的消息裁剪到最近 MAX_MESSAGES 条。"""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, message.role, message.content, datetime.now().isoformat()),
        )
        conn.execute(
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
            (session_id, session_id, MAX_MESSAGES),
        )


def get_history_str(session_id: str) -> str:
    """最近 HISTORY_WINDOW 条消息拼成提示词里的历史段落。"""
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

    history = ""
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
    """清空消息但保留会话本身；会话不存在返回 False。

    保留会话行是有意的：原实现里 delete 之后会话仍然存在，再查历史会返回空列表而不是 404。
    """
    with _connect() as conn:
        if not conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone():
            return False

        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    return True