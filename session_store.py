"""会话持久化：用 Redis 承载当前会话的最近消息与滚动摘要。

原先用本地 SQLite 文件，是为了解决「uvicorn --workers 2 时各进程各存一份、进程重启即丢」。
换成 Redis 后这两点依然满足，并且多了一层：API 进程与 worker 进程不再需要共享同一个
文件系统，Redis 是真正的共享存储；会话与记忆（memory/episodic、memory/profile）同源，
过期、容量、备份只有一套策略。

定位：会话是**正确性依赖**，不是缓存。缓存/限流/队列拿不到 Redis 可以降级，
会话不行 —— 降级只会把「读不到历史」伪装成「这是一段新对话」，属于静默的数据错误。
因此这里取连接用 cache.require_client()，拿不到就抛错。

键设计（前缀复用 cache.key()，与缓存同一版本号）：

    sess:{id}:meta   HASH    created_at / updated_at / next_id / user_id / title
    sess:{id}:msgs   LIST    元素 = JSON {"id", "role", "content", "timestamp"}
    sess:{id}:sum    STRING  滚动摘要

    另有按用户名下的会话索引，供前端侧边栏列出「我的历史会话」（原先只能靠手动输入
    session_id 才能进入某段对话）：

    sess:user:{digest(user_id)}:index   ZSET   member = session_id，score = 最近活跃时间

    索引只在带 user_id 的真实用户链路上维护；索引里的成员可能先于会话过期，list_sessions
    以 meta 是否存在为准过滤，并顺手把残留成员清出索引。

与 SQLite 版的语义对应（函数签名与行为全部保持一致，调用方零改动）：

    sessions 表的存在性        → meta 键是否存在
    messages 的自增 id         → meta 里的 next_id（HINCRBY）
    DELETE ... WHERE id <= max → 从左端弹出所有 id <= max_id 的元素
    ON DELETE CASCADE          → 显式 DEL 各个键

这里是两层记忆里「工作记忆」的落点，工作记忆的压缩由 context_compact.ContextCompactor
在 agent_loop 里完成（见 context_compact.py / agent.py）：
- msgs 保存当前会话的消息，agent 每轮开始时整批加载作为初始上下文；
- sum 保存一段滚动摘要，agent 侧压缩历史后由 save_compaction 把摘要写回这里，
  并删掉已被摘要覆盖的那批消息。

Lua 的两处必要性：
- 追加消息要「取号 → 入队 → 裁剪 → 续期」四步，拆开会有两个 worker 拿到同一号、
  或消息顺序与 id 顺序不一致的竞态；
- 提交压缩要「写摘要 → 删消息 → 续期」原子完成，否则会出现摘要更新了但消息没删，
  或消息删了但摘要没写的中间态。
"""

import json
import uuid
from datetime import datetime

import cache
from config import AGENT_HISTORY_LIMIT, SESSION_TTL_SECONDS
from models import ChatMessage
from observability import logger

# 消息条数的兜底硬上限：正常路径不该触发（agent 的上下文压缩会先把消息降下来），
# 一旦触发说明压缩流程失效，这里保证消息不会无界增长，同时打 warning 暴露问题。
MAX_MESSAGES_HARD = 200

_PURPOSE = "会话存储"

# 追加一条消息：取号 + 入队 + 按硬上限裁剪 + 续期。返回被裁掉的条数。
# 消息 JSON 在 Lua 里用 cjson 拼，就是为了让 id 与入队顺序在同一次原子执行里确定。
_ADD_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('HSET', KEYS[1], 'created_at', ARGV[1], 'next_id', 0)
end
local id = redis.call('HINCRBY', KEYS[1], 'next_id', 1)
redis.call('HSET', KEYS[1], 'updated_at', ARGV[1])

-- 首条用户消息截断后作为会话标题（标题已写定就不再改）。截断在 Python 侧做，
-- 这里不做 string.sub：按字节截断会把中文切到半个字符，落成非法 UTF-8。
if ARGV[2] == 'user' and ARGV[7] ~= '' then
    local title = redis.call('HGET', KEYS[1], 'title')
    if not title or title == '' then
        redis.call('HSET', KEYS[1], 'title', ARGV[7])
    end
end

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

# 写回压缩结果：弹出所有被摘要覆盖的消息 + 写摘要 + 续期，一次原子完成。
# 用「弹出左端 id <= max_id 的元素」而不是「删前 N 条」：max_id 是被摘要覆盖的那批
# 消息里最大的 id，压缩期间新插入的消息 id 更大，无论它是否已经排到前面都不会被误删。
_SAVE_SUMMARY_LUA = """
local max_id = tonumber(ARGV[1])
while true do
    local head = redis.call('LINDEX', KEYS[2], 0)
    if not head then break end
    local ok, item = pcall(cjson.decode, head)
    if not ok or tonumber(item['id']) == nil or tonumber(item['id']) > max_id then break end
    redis.call('LPOP', KEYS[2])
end

redis.call('SET', KEYS[3], ARGV[2], 'EX', ARGV[3])
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


def _user_index_key(user_id: str) -> str:
    """该用户名下会话索引的键名。

    用 digest 把 user_id 压成定长十六进制：user_id 是前端生成、可能含冒号等分隔符的
    自由文本，直接拼进键名会与键本身「: 分隔」的结构混淆。
    """
    return cache.key("sess", "user", cache.digest(user_id), "index")


def _index_session(client, session_id: str, user_id: str, score: float) -> None:
    """把会话挂到该用户名下的索引，并把分值刷成最近活跃时间，同时续期。

    user_id 为空（评测脚本、裸调用）时不建索引——侧边栏只服务真实用户。
    对已存在但先前没带 user_id 的会话，这里相当于补挂一次。
    """
    if not user_id:
        return
    index_key = _user_index_key(user_id)
    pipe = client.pipeline(transaction=False)
    pipe.zadd(index_key, {session_id: score})
    pipe.expire(index_key, SESSION_TTL_SECONDS)
    pipe.execute()


def _session_title(message: ChatMessage) -> str:
    """会话标题：首条用户消息折叠空白后截前 40 字；非用户消息返回空串（Lua 侧据此不写）。

    截断放在 Python 侧，因为 message.content 是已解码的 str，按字符切片不会把中文
    劈成半个字符（Redis 的 Lua string.sub 是按字节的）。
    """
    if message.role != "user":
        return ""
    return " ".join((message.content or "").split())[:40]


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


def get_or_create_session(session_id: str | None, user_id: str = "") -> str:
    """session_id 已存在就用它，否则新建一个并返回。

    取到已有会话时顺手续一次 TTL：每次对话开头都会调这里，于是「活跃会话不过期」。
    user_id 非空时把会话挂到该用户名下的索引（分值 = 本次活跃时间），供侧边栏列会话；
    没有 user_id 的调用方（评测、裸 curl）不建索引。
    """
    client = _client()
    user_id = (user_id or "").strip()
    now = datetime.now()
    if session_id and client.exists(_key(session_id, "meta")):
        _touch(client, session_id)
        _index_session(client, session_id, user_id, now.timestamp())
        return session_id

    session_id = new_session_id()
    stamp = now.isoformat()
    pipe = client.pipeline(transaction=False)
    pipe.hset(
        _key(session_id, "meta"),
        mapping={
            "created_at": stamp,
            "updated_at": stamp,
            "next_id": 0,
            "user_id": user_id,
            "title": "",
        },
    )
    for suffix in ("meta", "msgs", "sum"):
        pipe.expire(_key(session_id, suffix), SESSION_TTL_SECONDS)
    pipe.execute()
    _index_session(client, session_id, user_id, now.timestamp())
    return session_id


def add_message(session_id: str, message: ChatMessage) -> None:
    """追加一条消息，并只在超过兜底硬上限时裁剪。

    正常的瘦身交给 agent_loop 里的上下文压缩（context_compact.ContextCompactor，
    压缩后由 save_compaction 把摘要写回并删除被覆盖的消息），这里只是最后一道防线：
    真触发了说明压缩没跑起来，打 warning 暴露。
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
            _session_title(message),
        ],
    )

    if int(trimmed or 0):
        logger.warning(
            "messages.hard_trim",
            extra={"fields": {"session_id": session_id, "trimmed": int(trimmed)}},
        )


def get_summary(session_id: str) -> str | None:
    """当前会话的滚动摘要；没有或为空时返回 None。"""
    raw = _client().get(_key(session_id, "sum"))
    if raw is None:
        return None
    summary = _decode(raw).strip()
    return summary or None


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

    保留 meta 是有意的：删历史之后会话仍然存在，再查历史会返回空列表而不是 404。
    摘要必须一起清掉，否则「清空历史」之后助手仍然记得之前聊过什么。
    """
    client = _client()
    if not client.exists(_key(session_id, "meta")):
        return False

    client.delete(
        _key(session_id, "msgs"),
        _key(session_id, "sum"),
    )
    return True


def list_sessions(user_id: str, number: int = 20) -> list[dict]:
    """按最近活跃倒序列出该用户名下的会话，供前端侧边栏渲染。

    返回 [{"session_id", "title", "created_at", "updated_at"}]，最新的在前；
    user_id 为空（评测等无身份调用）时返回空列表。

    索引成员可能已经先于会话过期（三个会话键到 TTL 被删、索引成员还在），
    这里以 meta 是否存在为准过滤，并把残留成员顺手从索引里清掉，避免越积越多。
    """
    client = _client()
    user_id = (user_id or "").strip()
    if not user_id:
        return []

    index_key = _user_index_key(user_id)
    session_ids = [
        _decode(raw) for raw in client.zrevrange(index_key, 0, max(number, 1) - 1)
    ]
    if not session_ids:
        return []

    # 一次 pipeline 把 N 个 meta 拿回来，避免侧边栏列出 20 条就打 20 次往返
    pipe = client.pipeline(transaction=False)
    for session_id in session_ids:
        pipe.hgetall(_key(session_id, "meta"))
    metas = pipe.execute()

    sessions = []
    stale = []
    for session_id, raw_meta in zip(session_ids, metas):
        if not raw_meta:
            stale.append(session_id)
            continue
        meta = {_decode(key): _decode(value) for key, value in raw_meta.items()}
        sessions.append(
            {
                "session_id": session_id,
                "title": meta.get("title") or "",
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
            }
        )

    if stale:
        client.zrem(index_key, *stale)

    return sessions


def load_context(session_id: str, limit: int = AGENT_HISTORY_LIMIT) -> dict:
    """加载 agent 的初始上下文。

    返回 {"messages", "summary", "max_id"}：
    - messages：最近 limit 条 user / assistant 消息，正序，元素是 OpenAI 格式的
      {"role", "content"}；若会话已有滚动摘要，则在最前面插入一条承载摘要的 user 消息。
    - summary：当前滚动摘要（无则 None），供调用方判断是否需要重新压缩。
    - max_id：这批消息里最大的 id；压缩提交时按「id <= max_id」删除被摘要覆盖的那批。

    默认 limit 与 MAX_MESSAGES_HARD 对齐，即一次把现存消息全部加载进来，
    避免「截断后按 max_id 删除」误删没进摘要的旧消息。
    """
    client = _client()
    raw = client.lrange(_key(session_id, "msgs"), -max(limit, 1), -1)
    items = [_item(item) for item in raw]
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in items
        if item["role"] in ("user", "assistant")
    ]
    summary = get_summary(session_id)
    if summary:
        messages.insert(0, {"role": "user", "content": f"[此前对话摘要]\n{summary}"})
    max_id = max((int(item["id"]) for item in items if item.get("id") is not None), default=0)
    return {"messages": messages, "summary": summary, "max_id": max_id}


def save_compaction(session_id: str, summary: str, max_id: int) -> None:
    """把压缩结果写回会话：摘要落到 sum，删掉 id <= max_id 的已被覆盖消息。

    「写摘要 + 删消息 + 续期」在同一个 Lua 脚本里原子完成，避免出现
    「摘要更新了但消息没删」或「消息删了但摘要没写」的中间态。
    """
    client = _client()
    _script(client, "save_summary", _SAVE_SUMMARY_LUA)(
        keys=[
            _key(session_id, "meta"),
            _key(session_id, "msgs"),
            _key(session_id, "sum"),
        ],
        args=[max_id, summary, SESSION_TTL_SECONDS],
    )