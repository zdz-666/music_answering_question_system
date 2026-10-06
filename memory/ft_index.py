"""记忆的向量索引：情景记忆与用户画像的存储 / 检索，落在 Redis 自带的 RediSearch 上。

原先两层记忆各用一个 Chroma 集合。改用 Redis 之后，记忆与会话同源：
同一个实例、同一套过期与备份策略，API 进程与 worker 进程不必再共享一个文件系统；
也让「会话与记忆以 Redis 为准」这件事只有一处配置（REDIS_URL）。

为什么用 FT.*（RediSearch）而不是自建索引：
- 一份 hash 同时存正文、时间戳与向量，检索时一次往返就把命中的正文取回来，
  不需要再维护一个「元素 id → 属性」的旁路结构，也就没有两者的同步问题；
- 向量按 float32 原样存、KNN 直接给余弦距离，没有量化误差；
- 按 uid 做 TAG 过滤是服务端行为，别人的记忆根本不会进入候选集。

键与索引布局（前缀复用 cache.key()，与缓存 / 会话同一版本号）：

    musicrag:v1:memidx:{collection}           索引名
    musicrag:v1:memdoc:{collection}:{doc_id}  一份记忆一个 hash
        uid      TAG      归属用户（md5 摘要，用于过滤）
        sid      TAG      来源会话（md5 摘要，用于排除当前会话）
        user     文本     原始 user_id，仅供排查与读回，不建索引
        session  文本     原始 session_id，读回给调用方，不建索引
        text     文本     正文，不建索引，只用于读回
        ts       文本     写入时间，不建索引，只用于读回
        vec      VECTOR   float32 余弦向量

    sid 存摘要、session 存原文是有意分开的：TAG 只用来匹配，返回给调用方的必须是
    原始 session_id —— 否则调用方拿它当 exclude_session_id 回传时会被再摘要一次，
    过滤条件就再也匹配不上（这个坑实测踩过）。

两个刻意的取舍：
- **TAG 值一律先 md5 再进索引**。RediSearch 的 TAG 查询要转义 `{ } $ \\ |` 等字符、
  并按分隔符切分，而 user_id / session_id 都来自客户端（`/api/chat` 直接收 JSON 字段，
  可以是任意字符串）。先摘要就没有任何需要转义的字符，索引与查询两侧永远一致；
  原始的 user_id 另存一个 `user` 字段，排查时不至于看不出来是谁的。
- **索引维度由第一次写入的向量决定**（FT.CREATE 的 DIM 取 len(vector)），
  这样不会出现「配置里写 4096、模型却换成 1024 维」的错配。代价是换嵌入模型后
  旧索引与新向量不兼容：需要 FT.DROPINDEX 并清掉该集合的 memdoc:* 后重新积累，
  README 里写明了这条。
"""

import hashlib
import struct

import cache

# hash 里的字段名，同时也是 FT.SEARCH 的 RETURN 目标
FIELD_UID = "uid"
FIELD_SID = "sid"
FIELD_USER = "user"
FIELD_SESSION = "session"
FIELD_TEXT = "text"
FIELD_TS = "ts"
FIELD_VEC = "vec"

# 读回给调用方的字段（不含向量，也不含只用于过滤的 TAG）
_RETURN_FIELDS = (FIELD_TEXT, FIELD_TS, FIELD_SESSION, FIELD_USER, "score")

# Redis 在索引已存在时返回的固定文案（SEARCH_INDEX_EXISTS Index already exists）
_ALREADY_EXISTS = "already exists"
# 索引还没建过时 FT.SEARCH 的返回（SEARCH_INDEX_NOT_FOUND Index not found: xxx）
_INDEX_MISSING = "index not found"


def _client():
    # 记忆是数据本身，不是缓存：拿不到 Redis 就抛错，由 memory 层记 warning 并跳过记忆
    return cache.require_client("记忆存储")


def tag_id(value: str) -> str:
    """TAG 字段值：先做摘要，避开 RediSearch 的转义与分隔符规则。"""
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _pack(vector) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def index_name(collection: str) -> str:
    return cache.key("memidx", collection)


def doc_prefix(collection: str) -> str:
    return cache.key("memdoc", collection) + ":"


def doc_key(collection: str, doc_id: str) -> str:
    return cache.key("memdoc", collection, doc_id)


def ensure_index(collection: str, dim: int) -> None:
    """建索引，幂等。

    索引定义会随 AOF/RDB 存活，但这里刻意**不做进程内的「已建过」缓存**：
    一旦有人外部 FT.DROPINDEX、或换了数据卷，缓存住的标记会让后续写入不再建索引
    （数据写进去了、却搜不出来），这种沉默的错误很难发现。改成每次写入前都发一条
    FT.CREATE，已存在时 Redis 直接回 SEARCH_INDEX_EXISTS，代价只是一次往返。
    """
    client = _client()
    try:
        client.execute_command(
            "FT.CREATE", index_name(collection), "ON", "HASH",
            "PREFIX", 1, doc_prefix(collection),
            "SCHEMA",
            FIELD_UID, "TAG",
            FIELD_SID, "TAG",
            FIELD_VEC, "VECTOR", "HNSW", 6,
            "TYPE", "FLOAT32", "DIM", int(dim), "DISTANCE_METRIC", "COSINE",
        )
    except Exception as exc:  # noqa: BLE001 - 已存在是正常路径，其余要变成可读错误
        if _ALREADY_EXISTS in str(exc):
            return
        raise RuntimeError(
            f"记忆索引 {index_name(collection)} 创建失败：{type(exc).__name__}: {exc}"
        ) from exc


def upsert(
    collection: str,
    doc_id: str,
    vector,
    text: str,
    user_id: str,
    timestamp: str,
    session_id: str | None = None,
) -> None:
    """写一条记忆。doc_id 是内容的确定性摘要，重复写入即覆盖，天然幂等。"""
    client = _client()
    ensure_index(collection, len(vector))

    fields = {
        FIELD_UID: tag_id(user_id),
        FIELD_USER: user_id,
        FIELD_TEXT: text,
        FIELD_TS: timestamp,
        FIELD_VEC: _pack(vector),
    }
    if session_id:
        fields[FIELD_SID] = tag_id(session_id)
        fields[FIELD_SESSION] = session_id
    client.hset(doc_key(collection, doc_id), mapping=fields)


def _parse(raw) -> list[dict]:
    """把 FT.SEARCH 的返回摊平成 [{"id","text","ts","session","user","distance"}]。

    redis-py 会把结果解析成 {"results": [{"id", "extra_attributes"}, ...]}。
    结构不符合预期时直接抛错（由 memory 层记 warning），而不是当作「没有记忆」——
    后者会把解析问题伪装成「该用户还没有记忆」，正好是最难排查的那种。
    """
    if not raw:
        return []
    if not isinstance(raw, dict):
        raise RuntimeError(f"FT.SEARCH 返回了意外的结构：{type(raw).__name__}")

    results = raw.get(b"results") or raw.get("results") or []
    hits = []
    for item in results:
        extra = item.get(b"extra_attributes") or item.get("extra_attributes") or {}
        fields = {_decode(k): _decode(v) for k, v in extra.items()}
        hits.append(
            {
                "id": _decode(item.get(b"id") or item.get("id") or ""),
                "text": fields.get(FIELD_TEXT, ""),
                "ts": fields.get(FIELD_TS),
                "session": fields.get(FIELD_SESSION),
                "user": fields.get(FIELD_USER),
                "distance": float(fields.get("score") or 0.0),
            }
        )
    return hits


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def search(
    collection: str,
    vector,
    user_id: str,
    k: int,
    exclude_session_id: str | None = None,
) -> list[dict]:
    """取该用户语义最相近的 k 条记忆，按距离升序；排除指定会话。

    返回的 distance 是 RediSearch 的 COSINE 距离（= 1 - 余弦相似度），越小越相近；
    session 是**原始** session_id，可直接回传给 exclude_session_id。
    索引不存在说明这个用户还没有任何记忆，返回空列表即可（不是错误）。
    """
    if k <= 0:
        return []

    client = _client()
    query = f"(@{FIELD_UID}:{{{tag_id(user_id)}}}"
    if exclude_session_id:
        # TAG 取反：缺 sid 字段的文档（画像事实）不会被排除，符合预期
        query += f" -@{FIELD_SID}:{{{tag_id(exclude_session_id)}}}"
    query += f")=>[KNN {int(k)} @{FIELD_VEC} $q AS score]"

    try:
        raw = client.execute_command(
            "FT.SEARCH", index_name(collection), query,
            "PARAMS", 2, "q", _pack(vector),
            "SORTBY", "score",
            "RETURN", len(_RETURN_FIELDS), *_RETURN_FIELDS,
            "DIALECT", 2,
        )
    except Exception as exc:  # noqa: BLE001 - 只有「索引还没建」算正常空结果
        if _INDEX_MISSING in str(exc).lower():
            return []
        raise
    return _parse(raw)


def count(collection: str, user_id: str) -> int:
    """该用户在这个集合里有多少条记忆（验证脚本与排查用）。"""
    client = _client()
    try:
        raw = client.execute_command(
            "FT.SEARCH", index_name(collection),
            f"(@{FIELD_UID}:{{{tag_id(user_id)}}})",
            "LIMIT", 0, 0,
        )
    except Exception as exc:  # noqa: BLE001 - 索引还没建 = 0 条
        if _INDEX_MISSING in str(exc).lower():
            return 0
        raise

    if isinstance(raw, dict):
        return int(raw.get(b"total_results") or raw.get("total_results") or 0)
    # 老解析器的结构：[总数, key, [字段...], ...]
    return int(raw[0]) if raw else 0