"""基于 Redis 的令牌桶限流。

为什么必须是 Lua：一次限流要做「读令牌 → 按时间补令牌 → 扣令牌 → 写回」四步，
拆成多次 Redis 调用就会出竞态 —— 两个并发请求可能都读到「够用」的同一份令牌，
于是双双放行。Redis 单线程 + Lua 脚本原子执行，正好把这一串动作变成一次调用。

桶的状态存一个 hash：`tokens` 是剩余令牌，`ts` 是上次补充的时刻。补充是惰性的，
每次请求按距上次的时间差补 `elapsed × refill_per_sec`，补到容量上限为止。
相比固定窗口计数，令牌桶不会在窗口切换的瞬间被放行两倍流量；
相比每次都写一个带 TTL 的计数器，它也不用为突发流量额外留窗口状态。

Redis 不可用时一律**放行**：限流是保护自己的措施，不是正确性依赖。
保护层自己挂了却把正常请求也拒掉，是把可用性换成了理论上的安全，不划算。
"""

import time

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from config import (
    RATE_LIMIT_CHAT_CAPACITY,
    RATE_LIMIT_CHAT_REFILL_PER_SEC,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_UPLOAD_CAPACITY,
    RATE_LIMIT_UPLOAD_REFILL_PER_SEC,
)
from observability import logger

# 每个桶的 (容量, 每秒补充)。容量决定能突发多少次，补充速率决定长期平均速率。
BUCKETS = {
    "chat": (RATE_LIMIT_CHAT_CAPACITY, RATE_LIMIT_CHAT_REFILL_PER_SEC),
    "upload": (RATE_LIMIT_UPLOAD_CAPACITY, RATE_LIMIT_UPLOAD_REFILL_PER_SEC),
}

# 返回 {是否放行, 剩余令牌(向下取整), 还需等待的秒数}
# 令牌用浮点存，只有 >= 1 才扣；不足时按补充速率反推要等多久，写进 Retry-After。
_TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])

if tokens == nil then
    -- 第一次见到这个桶：装满它，让新用户/新 IP 不会一开始就被卡
    tokens = capacity
    ts = now
end

tokens = math.min(capacity, tokens + math.max(0, now - ts) / 1000 * refill)

local allowed = 0
local retry_after = 0
if tokens >= 1 then
    allowed = 1
    tokens = tokens - 1
else
    retry_after = math.ceil((1 - tokens) / refill)
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
-- 空闲到桶重新装满的时间，之后再没人来访问就让它自己过期，不占内存
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / refill * 1000) + 60000)

return {allowed, math.floor(tokens), retry_after}
"""

_scripts = {}


def _bucket_key(bucket: str, identity: str) -> str:
    # 这里刻意不 import cache.key，避免 ratelimit → cache → config 的绕行；
    # 复用同一个前缀即可，键名本身各成体系。
    from cache import key

    return key("rl", bucket, identity)


def _consume(bucket: str, identity: str) -> tuple:
    """真正扣令牌。返回 (allowed, remaining, retry_after)；Redis 不可用时全放行。"""
    from cache import get_client

    client = get_client()
    if client is None:
        return True, -1, 0

    capacity, refill = BUCKETS[bucket]
    script = _scripts.get(bucket)
    if script is None:
        script = client.register_script(_TOKEN_BUCKET_LUA)
        _scripts[bucket] = script

    try:
        allowed, remaining, retry_after = script(
            keys=[_bucket_key(bucket, identity)],
            args=[capacity, refill, int(time.time() * 1000)],
        )
    except Exception as exc:  # noqa: BLE001 - 限流不可用不该拦住业务
        logger.warning(
            "ratelimit.error",
            extra={"fields": {"bucket": bucket, "error": f"{type(exc).__name__}: {exc}"}},
        )
        return True, -1, 0

    return bool(int(allowed)), int(remaining), int(retry_after)


def _identity(request: Request, user_id: str | None) -> str:
    """限流主体：优先用前端存在 localStorage 的 user_id，没有就退回客户端 IP。

    没有 user_id 时必须退到 IP，否则任何人清空 localStorage 就能绕开限流。
    """
    if user_id and user_id.strip():
        return f"u:{user_id.strip()}"
    host = request.client.host if request.client else ""
    return f"ip:{host or 'unknown'}"


async def enforce(bucket: str, request: Request, user_id: str | None = None) -> None:
    """扣一个令牌；超限直接抛 429（带 Retry-After），Redis 不可用则放行。

    同步的 redis 客户端会阻塞事件循环，所以下沉线程池 —— 这是全项目统一的规矩。

    这里用显式调用而不是 FastAPI dependency：/api/chat 的 user_id 在 JSON body 里，
    /api/upload 的在 multipart form 里，两种来源要在依赖里分别声明 body 参数才能取到，
    反而比在端点里写一行 `await ratelimit.enforce(...)` 更绕。
    """
    if not RATE_LIMIT_ENABLED:
        return

    identity = _identity(request, user_id)
    allowed, remaining, retry_after = await run_in_threadpool(_consume, bucket, identity)
    if allowed:
        return

    logger.warning(
        "ratelimit.rejected",
        extra={
            "fields": {
                "bucket": bucket,
                "identity": identity,
                "retry_after": retry_after,
                "capacity": BUCKETS[bucket][0],
            }
        },
    )
    raise HTTPException(
        status_code=429,
        detail=f"请求过于频繁，请 {retry_after} 秒后再试",
        headers={"Retry-After": str(retry_after)},
    )