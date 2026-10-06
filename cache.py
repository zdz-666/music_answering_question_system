"""Redis 缓存层：把「同样的输入不必算第二遍」这件事收敛到一个模块里。

缓存只解决速度，不承担正确性——因此本模块的每条路径都遵循同一个约定：
未配置 REDIS_URL、连不上、或调用过程中出错，都静默退化成「没有缓存」，
把控制权交回调用方，业务链路照常跑完。

分层原则（越靠前的层输入越稳定，可以缓存得越久）：
- 嵌入向量：embed_query 是纯函数，同样的文本必然得到同样的向量 → CACHE_TTL_EMBEDDING
- 查询重写 / 集合路由：LLM 对单条输入的确定性映射 → CACHE_TTL_LLM
- 检索结果：只依赖「集合内容 + 改写后的查询」 → CACHE_TTL_RETRIEVAL
- 最终答案：还受历史、长期记忆、联网结果影响，只有这些全为空时才可缓存 → CACHE_TTL_ANSWER

失效策略用「版本号」而不是「扫键删除」：
- 全局版本在 REDIS_KEY_PREFIX 里，改版本号即整体作废；
- 知识库按集合维护一个自增版本号（kb_version / bump_kb_version），
  写入后旧键自然没人再拼得出来，剩下的靠 TTL 自行过期。
  这样既不用 SCAN（阻塞式遍历在大 keyspace 上有风险），也不会漏删。

值是 JSON 或 float32 二进制，不用 pickle：缓存是可丢弃的派生物，
不该在依赖升级后因为反序列化失败把整个进程带崩。
"""

import hashlib
import json
import struct

import redis
from langchain_core.embeddings import Embeddings

from config import (
    CACHE_TTL_EMBEDDING,
    REDIS_KEY_PREFIX,
    REDIS_URL,
)
from observability import logger

# 连接建立与读写的最长等待时间。缓存是加速手段，宁可放弃也不能把接口拖慢
_SOCKET_TIMEOUT = 2.0

# 只在第一次失败时告警，避免 Redis 挂掉后每个请求刷一行日志
_warned = False
# 命中/未命中计数，供验证脚本与压测读取（进程内统计，不落 Redis）
_hits = 0
_misses = 0

_client: redis.Redis | None = None


def _warn_once(event: str, detail: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    logger.warning(event, extra={"fields": {"detail": detail[:300]}})


def get_client() -> "redis.Redis | None":
    """返回可用的 Redis 客户端；不可用时返回 None。

    连接失败不缓存这个结果，下次调用会再试一次——本地 Redis 起来之后
    不重启进程就能自动恢复（连接被拒绝是立刻返回的，不会拖慢请求）。
    """
    global _client
    if _client is not None:
        return _client
    if not REDIS_URL:
        _warn_once("cache.disabled", "未配置 REDIS_URL，缓存已关闭")
        return None
    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            socket_timeout=_SOCKET_TIMEOUT,
            socket_connect_timeout=_SOCKET_TIMEOUT,
        )
        client.ping()
    except Exception as exc:  # noqa: BLE001 - 缓存不可用的原因很多，一律降级
        _warn_once("cache.unavailable", f"{type(exc).__name__}: {exc}")
        return None
    _client = client
    logger.info("cache.ready", extra={"fields": {"url": _mask_url(REDIS_URL)}})
    return _client


def require_client(purpose: str) -> "redis.Redis":
    """取「必需」的 Redis 客户端：拿不到连接就抛错，绝不降级。

    与 get_client() 的分工要分清楚：
    - get_client() 服务于缓存 / 限流 / 上传队列——这些是可选设施，拿不到就退化，
      业务照常跑完（每次现算、放行、退回同步处理）；
    - require_client() 服务于会话与记忆——数据本身就存在 Redis 里，
      没有 Redis 就没有数据可读可写，此时降级只会把「读不到历史」伪装成正常空结果，
      所以宁可直接报错，让问题在调用处显式暴露。

    connection 失败不写进 _client，因此 Redis 恢复后不需要重启进程。
    """
    client = get_client()
    if client is None:
        raise RuntimeError(
            f"{purpose}依赖 Redis：请在 .env 配置 REDIS_URL 并确保 Redis 可访问"
            "（启动命令见 README）。"
        )
    return client


def _mask_url(url: str) -> str:
    """日志里只留主机与端口，抹掉密码。"""
    head, _, tail = url.rpartition("@")
    return f"***@{tail}" if head else url


def key(*parts) -> str:
    """拼一个带版本前缀的键名。"""
    return ":".join([REDIS_KEY_PREFIX, *(str(part) for part in parts)])


def digest(*parts) -> str:
    """把任意输入压成定长摘要，作为键名的一部分。

    用 \\x1f（ASCII 单元分隔符）连接而不是逗号，避免 ("a","b") 与 ("a,b",) 撞成同一个键。
    """
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _run(operation, default):
    """执行一次 Redis 读/写，任何异常都退化成 default。"""
    client = get_client()
    if client is None:
        return default
    try:
        return operation(client)
    except Exception as exc:  # noqa: BLE001 - 运行期断连同样只降级
        _warn_once("cache.error", f"{type(exc).__name__}: {exc}")
        return default


def get_json(name: str):
    """读一个 JSON 值；未命中或缓存不可用时返回 None。"""
    global _hits, _misses
    raw = _run(lambda client: client.get(name), None)
    if raw is None:
        _misses += 1
        logger.debug("cache.miss", extra={"fields": {"key": name}})
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        # 写入方换了格式（比如改过 REDIS_KEY_PREFIX 之外的结构）：当作未命中
        _misses += 1
        return None
    _hits += 1
    logger.debug("cache.hit", extra={"fields": {"key": name}})
    return value


def set_json(name: str, value, ttl: int) -> None:
    """写一个 JSON 值，带过期时间。序列化失败说明这个值本就不该进缓存。"""
    try:
        raw = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "cache.serialize_failed",
            extra={"fields": {"key": name, "error": f"{type(exc).__name__}: {exc}"}},
        )
        return
    _run(lambda client: client.set(name, raw, ex=ttl), None)


def get_vector(name: str) -> list | None:
    """读一个 float32 向量；未命中返回 None。

    向量存二进制而不是 JSON：4096 维的 JSON 数组约 80KB，落成 float32 是 16KB，
    而且省掉了服务端与客户端两侧的数字解析。
    """
    global _hits, _misses
    raw = _run(lambda client: client.get(name), None)
    if raw is None:
        _misses += 1
        return None
    if not raw or len(raw) % 4:
        _misses += 1
        return None
    _hits += 1
    logger.debug("cache.hit", extra={"fields": {"key": name}})
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def set_vector(name: str, vector, ttl: int) -> None:
    raw = struct.pack(f"<{len(vector)}f", *vector)
    _run(lambda client: client.set(name, raw, ex=ttl), None)


def kb_version(collection: str) -> int:
    """集合当前的写入版本号。知识库每次写入自增，让该集合的旧检索缓存立即失联。"""
    raw = _run(lambda client: client.get(key("kbver", collection)), None)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def bump_kb_version(collection: str) -> None:
    """集合写入后调用：版本号 +1，旧检索缓存不再被命中。"""
    _run(lambda client: client.incr(key("kbver", collection)), None)


def stats() -> dict:
    """本进程的命中统计，用于验证缓存是否真的生效。"""
    total = _hits + _misses
    return {
        "hits": _hits,
        "misses": _misses,
        "hit_rate": round(_hits / total, 4) if total else 0.0,
    }


class CachedEmbeddings(Embeddings):
    """给任意 Embeddings 包一层查询缓存。

    只缓存 embed_query：它是纯函数，同样的文本必然得到同样的向量，且重复提问、
    补充检索、多集合轮询都会反复问同一个问题，命中率很高。
    embed_documents 是建库与写记忆时的批量调用，每段文本都不相同、调用次数有限，
    缓存只会白占内存，因此直接透传。
    """

    def __init__(self, inner: Embeddings):
        self._inner = inner

    def embed_query(self, text: str) -> list:
        name = key("emb", digest(text))
        cached = get_vector(name)
        if cached is not None:
            return cached
        vector = self._inner.embed_query(text)
        set_vector(name, vector, CACHE_TTL_EMBEDDING)
        return vector

    def embed_documents(self, texts: list) -> list:
        return self._inner.embed_documents(texts)