"""会话持久化：用 Redis 承载当前会话的最近消息与滚动摘要。

原先用本地 SQLite 文件，是为了解决「uvicorn --workers 2 时各进程各存一份、进程重启即丢」。
换成 Redis 后这两点依然满足，并且多了一层：API 进程与 worker 进程不再需要共享同一个
文件系统，Redis 是真正的共享存储；会话与记忆（memory/episodic、memory/profile）同源，
过期、容量、备份只有一套策略。

定位：会话是**正确性依赖**，不是缓存。缓存/限流/队列拿不到 Redis 可以降级，
会话不行 —— 降级只会把「读不到历史」伪装成「这是一段新对话」，属于静默的数据错误。
因此这里取连接用 cache.require_client()，拿不到就抛错。

键设计（前缀复用 cache.key()，与缓存同一版本号）：

    sess:{id}:meta   HASH    created_at / updated_at / next_id
    sess:{id}:msgs   LIST    元素 = JSON {"id", "role", "content", "timestamp"}
    sess:{id}:sum    STRING  滚动摘要
    sess:{id}:claim  STRING  压缩认领（SET NX PX，值 = 认领时刻）

与 SQLite 版的语义对应（函数签名与行为全部保持一致，调用方零改动）：

    sessions 表的存在性        → meta 键是否存在
    messages 的自增 id         → meta 里的 next_id（HINCRBY）
    DELETE ... WHERE id <= max → 从左端弹出所有 id <= max_id 的元素
    compression_claims UPSERT  → SET key val NX PX（抢占成功即拿到）
    ON DELETE CASCADE          → 显式 DEL 各个键

这里是三层记忆里第一层（工作记忆）的落点：
- msgs 保存当前会话的最近消息，注入 prompt 的是最近 HISTORY_WINDOW 条；
- sum 保存一段滚动摘要，消息条数超过 COMPRESS_TRIGGER 时由 memory/working.py
  把最旧一批消息合并进摘要并删掉，压缩走「claim → LLM → commit」两阶段，
  claim 键保证多 worker 下同一会话不会被重复压缩。

Lua 的两处必要性：
- 追加消息要「取号 → 入队 → 裁剪 → 续期」四步，拆开会有两个 worker 拿到同一号、
  或消息顺序与 id 顺序不一致的竞态；
- 提交压缩要「写摘要 → 删消息 → 释放认领」原子完成，否则会出现摘要更新了但消息没删，
  或消息删了但摘要没写的中间态。
"""

import json
import time
import uuid
from datetime import datetime

import cache
from config import SESSION_TTL_SECONDS
from models import ChatMessage
from observability import logger

HISTORY_WINDOW = 4  # 生成答案时注入的历史消息条数（与原实现一致）
COMPRESS_BATCH = 10  # 每轮压缩最少处理的增量消息数
COMPRESS_TRIGGER = HISTORY_WINDOW + COMPRESS_BATCH  # 超过这个条数就触发摘要压缩
# 消息条数的兜底硬上限：正常路径永远不该触发（压缩会先把消息降回 HISTORY_WINDOW），
# 一旦触发说明压缩流程失效，这里保证消息不会无界增长，同时打 warning 暴露问题。
MAX_MESSAGES_HARD = 200
CLAIM_STALE_SECONDS = 60  # 压缩认领的过期秒数，兜住压缩过程中进程崩溃

_PURPOSE = "会话存储"

# 追加一条消息：取号 + 入队 + 按硬上限裁剪 + 续期。返回被裁掉的条数。
# 消息 JSON 在 Lua 里用 cjson 拼，就是为了让 id 与入队顺序在同一次原子执行里确定。
_ADD_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('HSET', KEYS[1], 'created_at', ARGV[1], 'next_id', 0)
end
local id = redis.call('HINCRBY', KEYS[1], 'next_id', 1)
redis.call('HSET', KEYS[1], 'updated_at', ARGV[1])
redis.call('RPUSH', KEYS[2], cjson.encode({
    id = id, role = ARGV[2], content = ARGV[3], timestamp = ARGV[4]
}))

local trimmed = 0
local total = redis.call('LLEN', KEYS[2])
local hard = tonumber(ARGV[6])
if total > hard then
    trimmed = total - hard
    redis.call('LTRIM', KEYS[2], trimmed, -1)
end

redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[5])
redis.call('EXPIRE', KEYS[3], ARGV[5])
return trimmed
"""

# 提交压缩：弹出所有被摘要覆盖的消息 + 写摘要 + 释放认领，一次原子完成。
# 用「弹出左端 id <= max_id 的元素」而不是「删前 N 条」：max_id 是本次读到的批次里
# 最大的 id，压缩期间新插入的消息 id 更大，无论它是否已经排到前面都不会被误删。
_COMMIT_LUA = """
local max_id = tonumber(ARGV[1])
while true do
    local head = redis.call('LINDEX', KEYS[2], 0)
    if not head then break end
    local ok, item = pcall(cjson.decode, head)
    if not ok or tonumber(item['id']) == nil or tonumber(item['id']) > max_id then break end
    redis.call('LPOP', KEYS[2])
end

redis.call('SET', KEYS[3], ARGV[2], 'EX', ARGV[3])
redis.call('DEL', KEYS[4])
redis.call('EXPIRE', KEYS[1], ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[3])
return 1
"""

_scripts: dict = {}


def _script(client, name: str, source: str):
    """按名字缓存已注册的脚本对象；redis-py 内部走 EVALSHA，未命中会自动 EVAL。

    客户端实例一旦连上就不会再变（见 cache.get_client），所以脚本可以放心复用。
    """
    script = _scripts.get(name)
    if script is None:
        script = client.register_script(source)
        _scripts[name] = script
    return script


def _client():
    return cache.require_client(_PURPOSE)


def _key(session_id: str, suffix: str) -> str:
    return cache.key("sess", session_id, suffix)


def _touch(client, session_id: str) -> None:
    """把会话的三个键一起续到 SESSION_TTL_SECONDS，并成一次往返。"""
    pipe = client.pipeline(transaction=False)
    for suffix in ("meta", "msgs", "sum"):
        pipe.expire(_key(session_id, suffix), SESSION_TTL_SECONDS)
    pipe.execute()


def _decode(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _item(raw) -> dict:
    """列表元素（JSON 文本）→ {"id", "role", "content", "timestamp"}。"""
    data = json.loads(_decode(raw))
    return {
        "id": data.get("id"),
        "role": data.get("role"),
        "content": data.get("content"),
        "timestamp": data.get("timestamp"),
    }


def init_db() -> None:
    """启动自检：确认 Redis 可访问。

    取代原先的「建表 + 切 WAL」。幂等，每个 worker 启动时各跑一次。
    连不上直接抛错，让问题在启动阶段暴露，而不是等第一个请求进来才发现。
    """
    client = _client()
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - 启动自检要给出可读原因
        raise RuntimeError(
            f"会话存储无法连接 Redis：{type(exc).__name__}: {exc}"
        ) from exc


def new_session_id() -> str:
    """时间戳 + 随机后缀。

    用 len(sessions) 当后缀的话，多 worker 下各进程各自计数，同一秒内会算出相同的 id；
    换成随机后缀后不需要任何共享计数器。
    """
    return f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


def get_or_create_session(session_id: str | None) -> str:
    """session_id 已存在就用它，否则新建一个并返回。

    取到已有会话时顺手续一次 TTL：每次对话开头都会调这里，于是「活跃会话不过期」。
    """
    client = _client()
    if session_id and client.exists(_key(session_id, "meta")):
        _touch(client, session_id)
        return session_id

    session_id = new_session_id()
    now = datetime.now().isoformat()
    pipe = client.pipeline(transaction=False)
    pipe.hset(
        _key(session_id, "meta"),
        mapping={"created_at": now, "updated_at": now, "next_id": 0},
    )
    for suffix in ("meta", "msgs", "sum"):
        pipe.expire(_key(session_id, suffix), SESSION_TTL_SECONDS)
    pipe.execute()
    return session_id


def add_message(session_id: str, message: ChatMessage) -> None:
    """追加一条消息，并只在超过兜底硬上限时裁剪。

    正常的瘦身交给 memory/working.py 的摘要压缩（把最旧一批消息并入摘要后删除），
    这里只是最后一道防线：真触发了说明压缩没跑起来，打 warning 暴露。
    """
    client = _client()
    trimmed = _script(client, "add", _ADD_LUA)(
        keys=[
            _key(session_id, "meta"),
            _key(session_id, "msgs"),
            _key(session_id, "sum"),
        ],
        args=[
            datetime.now().isoformat(),
            message.role,
            message.content,
            message.timestamp or datetime.now().isoformat(),
            SESSION_TTL_SECONDS,
            MAX_MESSAGES_HARD,
        ],
    )

    if int(trimmed or 0):
        logger.warning(
            "messages.hard_trim",
            extra={"fields": {"session_id": session_id, "trimmed": int(trimmed)}},
        )


def count_messages(session_id: str) -> int:
    """会话当前的消息条数，用于判断是否该做摘要压缩。"""
    return int(_client().llen(_key(session_id, "msgs")))


def get_summary(session_id: str) -> str | None:
    """当前会话的滚动摘要；没有或为空时返回 None。"""
    raw = _client().get(_key(session_id, "sum"))
    if raw is None:
        return None
    summary = _decode(raw).strip()
    return summary or None


def list_oldest_messages(session_id: str, limit: int) -> list[dict]:
    """最早 limit 条消息，返回 [{"id", "role", "content"}]。

    带上 id 是为了压缩提交时能按 id 精确删除这一批（并发新插入的消息 id 更大，不受影响）。
    """
    if limit <= 0:
        # 注意别让 LRANGE 0 -1 变成「取全部」：limit<=0 就该是空批次
        return []
    raw = _client().lrange(_key(session_id, "msgs"), 0, limit - 1)
    return [_item(item) for item in raw]


def claim_compression(session_id: str) -> bool:
    """压缩两阶段的第一步：抢占该会话的压缩权。

    原先是 SQLite 上一条 UPSERT + rowcount 判断，这里用 SET NX PX 表达同一件事：
    NX 保证「已有未过期认领」时抢占失败，于是并发调用里只有一个能拿到 True；
    PX 让认领在 CLAIM_STALE_SECONDS 后自动失效，等价于原实现里「刷新过期认领」的分支，
    进程压缩中途崩溃也不会把这个会话永远锁死。
    """
    client = _client()
    acquired = client.set(
        _key(session_id, "claim"),
        str(time.time()),
        nx=True,
        px=CLAIM_STALE_SECONDS * 1000,
    )
    return bool(acquired)


def release_compression(session_id: str) -> None:
    """压缩失败时释放认领：消息保留，下一轮再试。"""
    _client().delete(_key(session_id, "claim"))


def commit_compression(session_id: str, summary: str, max_id: int) -> None:
    """压缩两阶段的第二步：写回摘要 + 删除已被摘要覆盖的那批消息 + 释放认领。

    三件事在同一个 Lua 脚本里原子完成，避免出现「摘要更新了但消息没删」或
    「消息删了但摘要没写」的中间态。删除按 `id <= max_id` 而非「前 N 条」，
    压缩期间新插入的消息 id 更大，不会被误删。
    """
    client = _client()
    _script(client, "commit", _COMMIT_LUA)(
        keys=[
            _key(session_id, "meta"),
            _key(session_id, "msgs"),
            _key(session_id, "sum"),
            _key(session_id, "claim"),
        ],
        args=[max_id, summary, SESSION_TTL_SECONDS],
    )


def get_history_str(session_id: str) -> str:
    """最近 HISTORY_WINDOW 条消息拼成提示词里的历史段落，前面带上滚动摘要。

    摘要代表已被裁掉的那些轮次，最近几条代表刚刚发生的事，两者拼起来才是完整的工作记忆。
    """
    summary = get_summary(session_id)
    raw = _client().lrange(_key(session_id, "msgs"), -HISTORY_WINDOW, -1)

    history = f"【此前对话摘要】{summary}\n" if summary else ""
    for item in raw:  # LRANGE 从右端取，已是时间正序，无需再翻转
        message = _item(item)
        if message["role"] == "user":
            history += f"用户: {message['content']}\n"
        elif message["role"] == "assistant":
            history += f"助手: {message['content']}\n"
    return history


def list_messages(session_id: str, number: int) -> list[ChatMessage] | None:
    """最近 number 条消息；会话不存在时返回 None，由调用方转 404。"""
    client = _client()
    if not client.exists(_key(session_id, "meta")):
        return None

    raw = client.lrange(_key(session_id, "msgs"), -number, -1)
    return [
        ChatMessage(
            role=item["role"],
            content=item["content"],
            timestamp=item["timestamp"],
        )
        for item in (_item(raw_item) for raw_item in raw)
    ]


def clear_messages(session_id: str) -> bool:
    """清空消息与摘要；会话不存在返回 False。

    保留 meta 是有意的：原实现里 delete 之后会话仍然存在，再查历史会返回空列表而不是 404。
    摘要必须一起清掉，否则「清空历史」之后助手仍然记得之前聊过什么；
    认领也一并清掉，否则残留的认领会让接下来 CLAIM_STALE_SECONDS 内的压缩被跳过。
    """
    client = _client()
    if not client.exists(_key(session_id, "meta")):
        return False

    client.delete(
        _key(session_id, "msgs"),
        _key(session_id, "sum"),
        _key(session_id, "claim"),
    )
    return True