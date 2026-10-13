from langchain_core.prompts import ChatPromptTemplate
import agent
import cache
import document_loader
from collection_router import get_router_collection
from models import ChatMessage, QueryRequest
from dynamic_chunk import SemanticChunker
from data_storage import (
    in_memory_similarity_search,
    load_vector_store,
    vector_similarity_search,
)
from session_store import add_message, get_or_create_session
from config import (
    CACHE_TTL_LLM,
    CACHE_TTL_RETRIEVAL,
    HTTP_RETRIES,
    RERANK_BASE_URL,
    RERANK_MODEL,
    RERANK_TIMEOUT,
    TAVILY_API_URL,
    TAVILY_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT,
    get_chat_model,
    get_rerank_api_key,
    get_tavily_api_key,
)
from observability import log_step, logger
import httpx
import math
from datetime import datetime
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class _RetryableServiceError(RuntimeError):
    """上游临时故障（超时 / 网络错误 / 5xx / 429），重试有意义。"""


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    logger.warning(
        "http.retry",
        extra={
            "fields": {
                "attempt": retry_state.attempt_number,
                "max_attempts": HTTP_RETRIES + 1,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        },
    )


@retry(
    stop=stop_after_attempt(HTTP_RETRIES + 1),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception_type(_RetryableServiceError),
    before_sleep=_log_retry,
    reraise=True,
)
def _post_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: float = 15.0,
) -> dict:
    """POST JSON：必带超时，失败按指数退避重试。

    只有超时、网络错误、5xx、429 会重试——4xx 是请求本身的问题（比如密钥错），重试无意义，
    直接抛出去，避免把 3 次尝试都浪费在一个必然失败的请求上。
    """
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise _RetryableServiceError(f"连接失败: {type(exc).__name__}: {exc}") from exc

    if response.status_code >= 500 or response.status_code == 429:
        raise _RetryableServiceError(f"HTTP {response.status_code}: {response.text[:200]}")

    if response.is_error:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    try:
        return response.json()
    except ValueError as exc:
        raise _RetryableServiceError(f"响应不是合法 JSON: {response.text[:200]}") from exc


def query_rewriting(query: str):
    llm = get_chat_model()
    prompt = """
你是一个专业的查询优化助手。你的任务是将用户输入的自然语言查询重写为更适合向量数据库检索的形式。

用户输入：{question}

重写规则：
1. 移除无关的礼貌用语、问候语和冗余词语
2. 保留核心意图和关键实体
3. 将口语化表达转换为正式的检索查询
4. 消除歧义，明确查询意图
5. 扩展相关同义词（可选，当有助于检索时）
6. 保持查询简洁，通常在5-15个词之间

示例：
输入："嘿，我想了解一下最近人工智能在医疗领域有哪些新进展，谢谢！"
输出："人工智能在医疗健康领域的最新研究进展和应用"

输入："那个，我不太确定怎么表达，就是机器学习模型训练的时候怎么防止过拟合啊？"
输出："机器学习模型防止过拟合的方法和技术"

注意：不要添加任何额外的解释，直接输出优化后的查询。
"""
    rewriting_query = ChatPromptTemplate.from_template(prompt)
    messages = rewriting_query.format_messages(question=query)

    # 重写结果只由这条 query 决定，与历史、记忆无关，因此按 query 缓存是安全的；
    # 一次请求里多集合轮询与补充检索会反复走到这里，命中率很高。
    name = cache.key("rewrite", cache.digest(query))
    cached = cache.get_json(name)
    with log_step(
        "query_rewriting", input=query, query_len=len(query), cached=cached is not None
    ) as span:
        if cached is None:
            cached = llm.invoke(messages).content
            cache.set_json(name, cached, CACHE_TTL_LLM)
        span.output = cached
    return cached


def web_rewriting(query: str):
    llm = get_chat_model()
    prompt = """
你是一个专业的“搜索查询净化与关键词提取”助手。你的任务是将用户凌乱、口语化、包含无关信息的原始查询，转化为简洁、精准、适合直接用于搜索引擎（如Google、Bing、百度）的关键词或短语。

处理规则：
*   **去除噪声**：剔除所有感叹词、语气词、与核心意图无关的冗余描述、人称代词（如“我”、“我的”）、以及“请问”、“有没有人知道”、“谢谢”等社交性用语。
*   **保留核心**：识别并保留查询中的核心实体（如人名、地名、产品名、事件）、核心动作/需求、以及关键的时空、属性限定词。
*   **结构化输出**：将提取出的关键词，以最相关、最必要的顺序，组合成**一个**简洁的搜索短语。通常采用“核心问题/对象 + 关键限定”的结构。避免输出多个不连贯的单词。
*   **同义合并**：如果用户用不同词语表达了相同意思，合并为最标准、最常用的那个词。
*   **隐含需求显性化**：如果用户的问题隐含了一个更通用的搜索意图，可以适度扩展。例如，“怎么让我的苹果手机不卡” 可以提取为 “iPhone 运行卡顿 解决方法”。
*   **绝对禁止**：不要添加任何解释性文字，只输出净化后的搜索短语本身。

**输入/输出格式:**
- 输入: [用户原始查询]
- 输出: [净化后的搜索关键词/短语]

**上下文示例 :**
- 输入: “哎呀，我昨天看的那部科幻电影，叫什么来着，就是有外星飞船和时空穿越的，特效特别牛，好像是2018年左右的？”
- 输出: “2018年 外星飞船 时空穿越 科幻电影 推荐”

- 输入: “大神求助！我的Win10电脑开机特别慢，要等好几分钟，怎么解决啊？急急急！”
- 输出: “Win10 开机速度慢 优化 解决方法”

- 输入: “谁能告诉我，去日本东京旅游，除了东京塔和迪士尼，还有哪些值得去的、不那么游客扎堆的地方？”
- 输出: “东京 小众 旅游景点 推荐 非热门”

- 输入: “最近想买蓝牙耳机，主要用来通勤听播客，希望降噪好、续航长，预算1000左右，求推荐！”
- 输出: “蓝牙耳机 推荐 降噪 长续航 播客 1000元预算”

**现在，请处理以下用户查询：**
输入: {question}
输出:
"""
    rewriting_query = ChatPromptTemplate.from_template(prompt)
    messages = rewriting_query.format_messages(question=query)
    with log_step("web_rewriting", input=query, query_len=len(query)) as span:
        result = llm.invoke(messages)
        span.output = result.content
    return result.content


def web_search(query: str) -> list:
    """Tavily 联网搜索，返回最多 TAVILY_MAX_RESULTS 条结果（list[dict]）。

    原先只取 results[0]，请求已经带回来的其余结果被丢掉，且命中空列表时
    会抛 IndexError。现在取回全部结果并做类型过滤，空结果退化为空列表 +
    一条警告，由下游的格式化函数渲染成“无网络检索结果”。
    """
    with log_step(
        "web_search",
        input=query,
        provider="tavily",
        max_results=TAVILY_MAX_RESULTS,
    ) as span:
        # 直接调 Tavily 的 HTTP 接口，省掉 SDK，超时与重试统一由 _post_json 兜住
        response = _post_json(
            TAVILY_API_URL,
            {
                "api_key": get_tavily_api_key(),
                "query": query,
                "max_results": TAVILY_MAX_RESULTS,
            },
            timeout=WEB_SEARCH_TIMEOUT,
        )
        results = [
            item for item in (response.get("results") or []) if isinstance(item, dict)
        ][:TAVILY_MAX_RESULTS]
        if not results:
            logger.warning("web_search.empty", extra={"fields": {"query": query}})
        span.output = results
    return results

def get_web_search(query):
        web_result = web_search(query)
        return web_result

def rerank(documents, query):
    documents = [doc.page_content for doc in documents]
    l = len(documents)

    num_to_extract = math.ceil(l * 0.5)
    headers = {
        "Authorization": f"Bearer {get_rerank_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
    "model": RERANK_MODEL,
    "query": query,
    "documents": documents
}
    with log_step(
        "rerank",
        input={"query": query, "documents": documents},
        model=RERANK_MODEL,
        docs_in=l,
        docs_keep=num_to_extract,
    ) as span:
        # 检索没命中任何文档时（集合为空、或改写后的查询没召回）不做请求，
        # 但这一步仍然进追踪，否则链路里会凭空少一环。
        result = []
        if l:
            try:
                text = _post_json(
                    RERANK_BASE_URL, payload, headers, timeout=RERANK_TIMEOUT
                )
            except Exception as exc:
                # 重排只是让排序更准的增强步骤，上游挂掉不该把整条问答打成 500。
                # 退化为混合检索的原顺序（EnsembleRetriever 的 RRF 排序）截断。
                result = _degrade_rerank(documents, num_to_extract, exc)
            else:
                results = text.get("results") or []

                for i, item in enumerate(results):
                    if i >= num_to_extract:
                        break
                    idx = item.get("index")

                    if isinstance(idx, int) and 0 <= idx < l:
                        result.append(documents[idx])

                if not result:
                    # 返回了 200 但结构对不上（网关错误页、字段改名等），同样退化为原序
                    result = _degrade_rerank(
                        documents,
                        num_to_extract,
                        f"响应中没有可用的 index（results={len(results)}）",
                    )

        span.output = result

    return result


def _degrade_rerank(documents, num_to_extract, reason):
    """重排不可用时按原顺序取前 N 条，并留下可检索的降级日志。"""
    logger.warning(
        "rerank.degraded",
        extra={
            "fields": {
                "step": "rerank",
                "reason": f"{reason}"[:300],
                "docs_keep": num_to_extract,
            }
        },
    )
    return documents[:num_to_extract]


def route_collections(question: str) -> list:
    """由 LLM 判断这个问题该落到哪几个集合。

    与检索拆开是因为补充检索要复用首轮的集合名，不能每轮重跑一次路由。
    路由同样只由 question 决定，所以按 question 缓存 —— 补充检索每轮都要问一次路由，
    没有缓存的话同一个问题会被重复判断两三次。
    """
    name = cache.key("route", cache.digest(question))
    cached = cache.get_json(name)
    with log_step(
        "collection_router", input=question, cached=cached is not None
    ) as span:
        if cached is None:
            cached = get_router_collection(question)
            cache.set_json(name, cached, CACHE_TTL_LLM)
        span.output = cached
    return cached


def _cached_vector_search(collection_name: str, query_change: str, k: int) -> list:
    """单个集合的「混合检索 + 重排」结果（chunk 文本列表），按 集合版本 + 改写后的查询 缓存。

    这一步是最贵的：一次 Chroma 稠密检索要算一次嵌入（已单独缓存），BM25 要把集合
    全量文档读出来重建索引，重排还要一次跨网请求。缓存的唯一外部依赖是集合内容，
    知识库写入时 bump_kb_version 会让旧键立刻失联，因此不会读到过期的检索结果。

    rerank 返回的已经是 list[str]（它在开头就把 Document 拆成了 page_content），
    所以这里直接缓字符串列表，不必再承载 Document 的元数据。
    """
    name = cache.key(
        "search",
        collection_name,
        cache.kb_version(collection_name),
        cache.digest(query_change, k),
    )
    cached = cache.get_json(name)

    with log_step(
        "vector_search",
        input=query_change,
        collection=collection_name,
        k=k,
        cached=cached is not None,
    ) as span:
        if cached is not None:
            chunks = cached
        else:
            vector_result = vector_similarity_search(collection_name, query_change, k=k)
            chunks = rerank(vector_result, query_change)
            cache.set_json(name, chunks, CACHE_TTL_RETRIEVAL)
        span.output = chunks

    return chunks


def get_vector_search(query, collection_list, external_docs=None) -> list:
    """在给定集合列表上做混合检索 + 重排，返回重排后的 chunk 列表。

    返回 list[str]（而不是拼好的大字符串）是为了让多轮检索结果能按 chunk 粒度去重。
    external_docs 非 None 时改在调用方给的这批文本里检索（评测场景），忽略 collection_list；
    这批语料每次调用都不同且只在本次有效，因此不走缓存。
    """
    result = []

    if external_docs is not None:
        with log_step(
            "vector_search",
            input=query,
            corpus="external",
            docs=len(external_docs),
            k=6,
        ) as span:
            vector_result = in_memory_similarity_search(external_docs, query, k=6)
            # Document 对象直接进日志会是 <Document ...> 这样的 repr，这里只取正文
            span.output = [doc.page_content for doc in vector_result]
        return rerank(vector_result, query)

    for co_name in collection_list:
        result.extend(_cached_vector_search(co_name, query, 6))

    return result


def format_kb_chunks(chunks: list) -> str:
    """知识库 chunk 列表 → 交给 prompt 的字符串。

    结果只作为模板变量传入 ChatPromptTemplate，绝不拼进模板本身——
    语料里的 `{}`（上传的 XML、歌词）否则会被当成占位符解析。
    """
    if not chunks:
        return "无知识库检索结果"
    return "\n".join(chunks)


def format_web_items(items: list) -> str:
    """Tavily 条目列表 → 带序号的文本块（agent 的 web_search 工具与评测链路都复用）。"""
    if not items:
        return "无网络检索结果"
    blocks = []
    for i, item in enumerate(items, 1):
        title = (item.get("title") or "无标题").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        blocks.append(f"[{i}] {title}\nURL：{url}\n内容：{content}")
    return "\n\n".join(blocks)


def get_result(query: QueryRequest, session_id: str) -> str:
    """问答入口：交给 agent_loop（工具调用 + 记忆 + 上下文压缩），返回答案文本。

    原先这里自己拼 prompt、把历史与记忆当模板变量塞进去；现在这些都下沉到 agent：
    历史由 agent 从会话加载，记忆注入 system prompt，检索变成工具，上下文超限自动压缩。
    评测链路不走这里（已整体迁到 rag_pipeline.py，只服务离线评测）。
    """
    return agent.run(
        query.question,
        session_id=session_id,
        user_id=(query.user_id or "").strip(),
        file_content=query.file_content,
    )


def answer_with_file(
    filename: str,
    content: bytes,
    question: str | None,
    session_id: str | None,
    user_id: str,
) -> dict:
    """上传文件的完整问答流程：解析 → agent 问答 → 落会话。

    这段编排原先写在 main.upload_file 里。上传改成异步任务后 worker 也要跑同一套流程，
    留在入口层就会变成一份逻辑两个副本（而「同步兜底」这条路两个入口都要用），
    因此下沉到这里：main 用 run_in_threadpool 调它，worker 直接调它。

    同步函数 —— 内部全是阻塞调用（PyMuPDF、VLM、Embedding、LLM），
    调用方负责放进线程池或放进 worker 进程。
    """
    with log_step(
        "document.parse",
        input={"file": filename, "bytes": len(content)},
    ) as parse_span:
        # format 不支持时抛 UnsupportedFormatError，交给调用方映射成 400
        blocks = document_loader.parse_document(filename, content)
        file_content = document_loader.blocks_to_text(blocks)
        parse_span.output = {
            "blocks": len(blocks),
            "images": sum(1 for block in blocks if block.kind == "image"),
            "chars": len(file_content),
        }

    session_id = get_or_create_session(session_id)
    answer = agent.run(
        question or "请分析上传的文件",
        session_id=session_id,
        user_id=(user_id or "").strip(),
        file_content=file_content,
    )

    add_message(
        session_id,
        ChatMessage(
            role="user",
            content=f"已上传文件: {filename}"
            + (f"\n问题: {question}" if question else ""),
        ),
    )
    add_message(session_id, ChatMessage(role="assistant", content=answer))

    return {
        "answer": answer,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }


def add_self_introduction(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = load_vector_store("self_introduction")
    vector_store.add_documents(docs)
    # 集合内容变了，旧的检索缓存必须立刻作废，否则同一个问题会一直答出旧文档
    cache.bump_kb_version("self_introduction")

def add_music_analysis(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = load_vector_store("music_analysis")
    vector_store.add_documents(docs)
    cache.bump_kb_version("music_analysis")

def add_music_list(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = load_vector_store("music_list")
    vector_store.add_documents(docs)
    cache.bump_kb_version("music_list")