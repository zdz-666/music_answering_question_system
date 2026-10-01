from langchain_core.prompts import ChatPromptTemplate
from collection_router import get_router_collection
from models import QueryRequest
from dynamic_chunk import SemanticChunker
from data_storage import vector_similarity_search, load_vector_store
from config import (
    HTTP_RETRIES,
    MAX_SUPPLEMENT_ROUNDS,
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
from retrieval_planner import judge_retrieval_sufficiency, plan_retrieval_channels
import httpx
import math
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
    with log_step("query_rewriting", input=query, query_len=len(query)) as span:
        result = llm.invoke(messages)
        span.output = result.content
    return result.content


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
        query_change = web_rewriting(query)
        web_result = web_search(query_change)
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
    """
    with log_step("collection_router", input=question) as span:
        collection_list = get_router_collection(question)
        span.output = collection_list
    return collection_list


def get_vector_search(query, collection_list) -> list:
    """在给定集合列表上做混合检索 + 重排，返回重排后的 chunk 列表。

    返回 list[str]（而不是拼好的大字符串）是为了让多轮检索结果能按 chunk 粒度去重。
    """
    result = []
    query_change = query_rewriting(query)
    for co_name in collection_list:
        with log_step(
            "vector_search", input=query_change, collection=co_name, k=6
        ) as span:
            vector_result = vector_similarity_search(co_name, query_change, k=6)
            # Document 对象直接进日志会是 <Document ...> 这样的 repr，这里只取正文
            span.output = [doc.page_content for doc in vector_result]

        result.extend(rerank(vector_result, query_change))

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
    """Tavily 条目列表 → 带序号的文本块，供 self_reflection 的 {web_result} 使用。"""
    if not items:
        return "无网络检索结果"
    blocks = []
    for i, item in enumerate(items, 1):
        title = (item.get("title") or "无标题").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        blocks.append(f"[{i}] {title}\nURL：{url}\n内容：{content}")
    return "\n\n".join(blocks)


def _normalize_query(text: str) -> str:
    """折叠空白并小写，用于比较两条检索语句/两段文本是否重复。"""
    return " ".join((text or "").split()).lower()


def merge_kb_chunks(existing: list, new: list) -> list:
    """把新一轮的 chunk 追加到已有结果后，按规范化文本精确去重（保留首次出现顺序）。"""
    merged = list(existing)
    seen = {_normalize_query(chunk) for chunk in existing}
    for chunk in new:
        key = _normalize_query(chunk)
        if key and key not in seen:
            seen.add(key)
            merged.append(chunk)
    return merged


def _web_key(item: dict) -> str:
    """网络条目的去重键：优先 URL，退回标题，再退回内容前 120 字。"""
    if item.get("url"):
        return item["url"].strip().lower()
    text = item.get("title") or item.get("content") or ""
    return _normalize_query(text)[:120]


def merge_web_items(existing: list, new: list) -> list:
    """把新一轮的网络结果追加到已有结果后，按 URL/标题精确去重。"""
    merged = list(existing)
    seen = {_web_key(item) for item in existing}
    for item in new:
        key = _web_key(item)
        if key and key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _channel_label(uses_kb: bool, uses_web: bool) -> str:
    if uses_kb and uses_web:
        return "knowledge_base+web_search"
    return "knowledge_base" if uses_kb else "web_search"


def _supplement_retrieval(question, uses_kb, uses_web, collection_list, kb_chunks, web_items):
    """信息不足时用新语句再检索，最多 MAX_SUPPLEMENT_ROUNDS 轮。

    通道沿用首轮决策，集合名沿用首轮结果。任何异常都只终止补检索、
    保留已有结果继续往下走（上游抖动不该把整条问答打成 500）。
    """
    seen_queries = {_normalize_query(question)}
    for round_index in range(1, MAX_SUPPLEMENT_ROUNDS + 1):
        try:
            assessment = judge_retrieval_sufficiency(
                question=question,
                kb_text=format_kb_chunks(kb_chunks),
                web_text=format_web_items(web_items),
                channel=_channel_label(uses_kb, uses_web),
                round_index=round_index,
            )
        except Exception as exc:
            logger.warning(
                "supplement.assess_failed",
                extra={
                    "fields": {
                        "round": round_index,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                },
            )
            break

        if assessment.is_sufficient:
            break

        new_query = (assessment.new_query or "").strip()
        new_key = _normalize_query(new_query)
        if not new_key or new_key in seen_queries:
            # 语句为空或与已用过的重复：没有有效进展，再查也是白花一次调用
            break
        seen_queries.add(new_key)

        try:
            if uses_kb:
                kb_chunks = merge_kb_chunks(
                    kb_chunks, get_vector_search(new_query, collection_list)
                )
            if uses_web:
                web_items = merge_web_items(web_items, get_web_search(new_query))
        except Exception as exc:
            logger.warning(
                "supplement.retrieve_failed",
                extra={
                    "fields": {
                        "round": round_index,
                        "query": new_query,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                },
            )
            break

    return kb_chunks, web_items


def _retrieve_and_reflect(question: str) -> str:
    """通道规划 → 首轮检索 → 补充检索 → 自反思，返回可直接作为 {context} 的字符串。

    通道规划必须放在最前面：它决定 collection_router / web_rewriting 要不要跑，
    放晚了关掉的通道会被白跑一遍。
    """
    plan = plan_retrieval_channels(question)

    if not plan.use_knowledge_base and not plan.use_web_search:
        # 模型判定无需检索：跳过检索与自反思，让生成步骤凭自身知识作答
        return "本次提问无需检索，请基于你自身的音乐知识作答。"

    collection_list = []
    kb_chunks = []
    web_items = []
    if plan.use_knowledge_base:
        collection_list = route_collections(question)
        kb_chunks = get_vector_search(question, collection_list)
    if plan.use_web_search:
        web_items = get_web_search(question)

    kb_chunks, web_items = _supplement_retrieval(
        question,
        plan.use_knowledge_base,
        plan.use_web_search,
        collection_list,
        kb_chunks,
        web_items,
    )

    return self_reflection(
        question, format_kb_chunks(kb_chunks), format_web_items(web_items)
    )


def _generate(prompt_template: str, variables: dict, **step_fields):
    """格式化 prompt → 调 LLM → 写 span，返回 AIMessage（保持 get_result 的既有契约）。"""
    messages = ChatPromptTemplate.from_template(prompt_template).format_messages(**variables)

    llm = get_chat_model()
    with log_step("generate", input=variables, **step_fields) as span:
        result = llm.invoke(messages)
        span.output = result.content
    return result

def self_reflection(query, vector_result, web_result):
    llm = get_chat_model()
    prompt_template = """
        你是一个信息相关性过滤器。给定用户提问和在用户知识库与网络中检索到的信息，请从用户知识库、网络检索到的信息、历史对话记录中提炼出与用户提问相关的信息。如果没有相关信息，请返回“无相关信息”。
        用户提问:{query}
        知识库中检索到的信息：{vector_result}
        网络中检索到的信息：{web_result}
        请只返回知识库、网络检索到的信息、历史对话记录中与用户提问有关的信息。不要附加其他任何解释性内容。
        输出格式：
        相关的知识库信息：与用户提问有关的知识库信息
        相关的网络信息：与用户提问有关的网络信息
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.format_messages(query=query, vector_result=vector_result, web_result=web_result)
    with log_step(
        "self_reflection",
        input={
            "query": query,
            "vector_result": vector_result,
            "web_result": web_result,
        },
        has_vector=bool(vector_result),
        has_web=bool(web_result),
    ) as span:
        result = llm.invoke(messages)
        span.output = result.content
    logger.info("self_reflection.result", extra={"fields": {"content": result.content}})
    return result.content

def get_result(query: QueryRequest, history_string: str):
     # 检索通道（知识库 / 网络）改由 retrieval_planner 依据提问自动决定，
     # 不再读 query 上的开关字段。
     context = _retrieve_and_reflect(query.question)

     prompt_template = """
        你是一个专业的音乐知识问答助手，名为“乐典”。你的核心职责是准确、专业、清晰地回答用户关于音乐的一切问题。
        理解与分析：仔细分析用户输入的问题，明确其核心意图和所需的知识范畴。
        用户之前的聊天记录：{history_string}

        信息检索与整合：
        融合信息：将内部知识、网络搜索结果以及知识库信息进行智能比对、验证与融合，形成完整的答案。

        组织与输出：你的回答应当结构清晰、重点突出、语言友好。
        输出格式要求（必须严格遵守，前端会按 Markdown 渲染）：
        - 始终使用 Markdown 格式组织回答，不要输出大段无结构的纯文本。
        - 小节标题使用 ## 或 ###，不要使用一级标题 #。
        - 并列的内容使用无序列表（- ），有先后顺序的内容使用有序列表（1. 2. 3.）。
        - 关键术语、曲名、作品名、人名使用 **加粗** 强调，但不要整段加粗。
        - 需要横向对比多项内容时，使用 Markdown 表格呈现。
        - 引用原文或他人观点时使用 > 引用块。
        - 直接输出 Markdown 正文本身，严禁把整个回答包裹在 ``` 代码块中。

        用户的问题是：{question}
        用户上传的文件内容是：{file_content}
        与用户提问相关的信息是：{context}
        """
     return _generate(
         prompt_template,
         {
             "question": query.question,
             "history_string": history_string,
             "file_content": query.file_content,
             "context": context,
         },
         has_file=bool(query.file_content),
     )

def add_self_introduction(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = load_vector_store("self_introduction")
    vector_store.add_documents(docs)

def add_music_analysis(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = load_vector_store("music_analysis")
    vector_store.add_documents(docs)

def add_music_list(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = load_vector_store("music_list")
    vector_store.add_documents(docs)

def get_result_evaluate(query: str):
     # 与 get_result 共用同一条「通道规划 → 检索 → 补充检索 → 自反思」链路，
     # 只有生成用的 prompt 不同（评测场景不带历史与上传文件）。
     context = _retrieve_and_reflect(query)

     prompt_template = """
        你是一个专业的音乐知识问答助手，名为“乐典”。你的核心职责是准确、专业、清晰地回答用户关于音乐的一切问题。
        理解与分析：仔细分析用户输入的问题，明确其核心意图和所需的知识范畴。

        信息检索与整合：
        融合信息：将内部知识、网络搜索结果以及知识库信息进行智能比对、验证与融合，形成完整的答案。

        组织与输出：你的回答应当结构清晰、重点突出、语言友好。
        输出格式要求（必须严格遵守，前端会按 Markdown 渲染）：
        - 始终使用 Markdown 格式组织回答，不要输出大段无结构的纯文本。
        - 小节标题使用 ## 或 ###，不要使用一级标题 #。
        - 并列的内容使用无序列表（- ），有先后顺序的内容使用有序列表（1. 2. 3.）。
        - 关键术语、曲名、作品名、人名使用 **加粗** 强调，但不要整段加粗。
        - 需要横向对比多项内容时，使用 Markdown 表格呈现。
        - 引用原文或他人观点时使用 > 引用块。
        - 直接输出 Markdown 正文本身，严禁把整个回答包裹在 ``` 代码块中。

        用户的问题是：{question}

        与用户提问相关的信息是：{context}
        """
     result = _generate(
         prompt_template,
         {"question": query, "context": context},
         step_variant="evaluate",
     )
     return context, result.content