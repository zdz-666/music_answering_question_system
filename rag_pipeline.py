"""旧 RAG 管道（检索-补充检索-自反思-生成）。

这是改造为 agent_loop 之前的在线问答链路，现在只服务于离线评测
（llm_ev.py、rgb_eval.py），不在服务链路中使用。
"""

from langchain_core.prompts import ChatPromptTemplate

from config import MAX_SUPPLEMENT_ROUNDS, get_chat_model
from observability import log_step, logger
from rag import (
    format_kb_chunks,
    format_web_items,
    get_vector_search,
    get_web_search,
    route_collections,
)
from retrieval_planner import judge_retrieval_sufficiency, plan_retrieval_channels


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


def _supplement_retrieval(
    question, uses_kb, uses_web, collection_list, kb_chunks, web_items, external_docs=None
):
    """信息不足时用新语句再检索，最多 MAX_SUPPLEMENT_ROUNDS 轮。

    通道沿用首轮决策，集合名沿用首轮结果。任何异常都只终止补检索、
    保留已有结果继续往下走（上游抖动不该把整条问答打成 500）。
    external_docs 非 None 时补检索同样落在调用方给的语料上（评测场景）。
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
                    kb_chunks,
                    get_vector_search(new_query, collection_list, external_docs=external_docs),
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


def _retrieve_and_reflect(question: str, external_docs: list | None = None) -> str:
    """通道规划 → 首轮检索 → 补充检索 → 自反思，返回可直接作为 {context} 的字符串。

    通道规划必须放在最前面：它决定 collection_router / web_rewriting 要不要跑，
    放晚了关掉的通道会被白跑一遍。

    external_docs 非 None 时进入评测模式：跳过通道规划、集合路由与联网搜索，
    只在调用方给的这批文本里检索；其后的重排、补充检索、自反思链路完全一致。
    """
    if external_docs is not None:
        kb_chunks = get_vector_search(question, [], external_docs=external_docs)
        kb_chunks, _ = _supplement_retrieval(
            question, True, False, [], kb_chunks, [], external_docs=external_docs
        )
        return self_reflection(question, format_kb_chunks(kb_chunks), format_web_items([]))

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
    """格式化 prompt → 调 LLM → 写 span，返回 AIMessage（供评测链路取 .content）。"""
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

def get_result_evaluate(query: str, external_docs: list | None = None):
     # 与 get_result 共用同一条「通道规划 → 检索 → 补充检索 → 自反思」链路，
     # 只有生成用的 prompt 不同（评测场景不带历史与上传文件）。
     # external_docs 非 None 时把检索语料换成调用方给的一批文本
     # （如 RGB 评测里的 positive/negative 文档），链路其余部分不变。
     context = _retrieve_and_reflect(query, external_docs=external_docs)

     prompt_template = """
       你是一个专业的知识问答助手。

约束条件（必须严格遵守）：
1. 你只能基于用户提供的【相关信息】来回答问题，严禁使用你自身的知识或外部信息。
2. 如果用户提供的信息中没有任何内容与问题相关，或信息不足以回答该问题，你必须明确拒绝回答，回复"根据提供的信息无法回答该问题"或类似表述，严禁编造答案。
3. 你的回答应当严格基于给定信息，简洁明了，不要做额外的解释。

用户的问题是：{question}

与用户提问相关的信息是：{context}
        """
     result = _generate(
         prompt_template,
         {"question": query, "context": context},
         step_variant="evaluate",
     )
     return context, result.content