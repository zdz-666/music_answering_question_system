from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from collection_router import get_router_collection
from main import QueryRequest
from dynamic_chunk import SemanticChunker
from data_storage import milvus_similarity_search, create_vector_store_loaded
import requests
import math


def init_llm():
    llm = ChatOpenAI(
    model="your model name",
    api_key="your apikey",
    base_url="your baseurl",
    temperature=0.1
)
    return llm

def query_rewriting(query: str):
    llm = init_llm()
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
    result = llm.invoke(messages)
    return result.content


def web_rewriting(query: str):
    llm = init_llm()
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
    result = llm.invoke(messages)
    return result.content


def web_search(query: str):
    tavily_client = TavilyClient(api_key="your apikey")
    response = tavily_client.search(query=query, max_results=5)
    return response.get("results")[0]

def get_web_search(query):
        query_change = web_rewriting(query)
        web_result = web_search(query_change)
        return web_result

def rerank(documents, query):
    documents = [doc.page_content for doc, score in documents]
    l = len(documents)
    num_to_extract = math.ceil(l * 0.5)

    if l == 0:
        return []

    api_url = "https://api.siliconflow.cn/v1/rerank"
    headers = {
        "Authorization": "Bearer your apikey",
        "Content-Type": "application/json",
    }
    payload = {
    "model": "Qwen/Qwen3-Reranker-8B",
    "query": query,
    "documents": documents
}
    response = requests.post(api_url, json=payload, headers=headers)
    text = response.json()
    results = text.get("results", [])

    docs = []
    for i, result in enumerate(results):
        if i >= num_to_extract:
            break
        idx = result.get("index")

        docs.append(documents[idx])

    return docs
def get_vector_search(query):
        collection_list = get_router_collection(query)
        result = []
        query_change = query_rewriting(query)
        for co_name in collection_list:
             vector_result = milvus_similarity_search(co_name, query_change, k=6)
             result_list = rerank(vector_result, query_change)
             combined_content = ""
             for i, doc in enumerate(result_list):
                if i > 0:  
                    combined_content += "\n"
                combined_content += doc
             result.append(combined_content)

        vector_result = ""
        for i, context in enumerate(result):
             if i > 0:  
                vector_result += "\n"
             vector_result += context
            
        return vector_result

def self_reflection(query, vector_result, web_result):
    llm = init_llm()
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
    result = llm.invoke(messages)
    print(f"self_reflection result: {result.content}")
    return result.content

def get_result(query: QueryRequest, history_string: str):
     web_search_result = None
     vector_search_result = None

     if query.use_web_search:
        web_search_result = get_web_search(query.question)
     else:
        web_search_result = "未使用网络搜索"

     if query.use_knowledge_base:
        vector_search_result = get_vector_search(query.question)
     else:
        vector_search_result = "未使用知识库"

     context = self_reflection(query.question, vector_search_result, web_search_result)

     prompt_template = """
        你是一个专业的音乐知识问答助手，名为“乐典”。你的核心职责是准确、专业、清晰地回答用户关于音乐的一切问题。
        理解与分析：仔细分析用户输入的问题，明确其核心意图和所需的知识范畴。
        用户之前的聊天记录：{history_string}

        信息检索与整合：
        融合信息：将内部知识、网络搜索结果以及知识库信息进行智能比对、验证与融合，形成完整的答案。

        组织与输出：你的回答应当结构清晰、重点突出、语言友好。

        用户的问题是：{question}
        用户上传的文件内容是：{file_content}
        与用户提问相关的信息是：{context}
        """
     prompt = ChatPromptTemplate.from_template(prompt_template)
     messages = prompt.format_messages(history_string=history_string,
                                       question=query.question, 
                                       file_content=query.file_content,
                                       context=context)
    
     llm = init_llm()
     result = llm.invoke(messages)
     return result

def add_self_introduction(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = create_vector_store_loaded("self_introduction")
    vector_store.add_documents(docs)

def add_music_analysis(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = create_vector_store_loaded("music_analysis")
    vector_store.add_documents(docs)

def add_music_list(text: str):
    docs = SemanticChunker(overlap_size=0, max_chunk_size=500).chunk_document(text)
    vector_store = create_vector_store_loaded("music_list")
    vector_store.add_documents(docs)

def get_result_evaluate(query: str,use_web_search: bool, use_knowledge_base: bool):
     web_search_result = None
     vector_search_result = None

     if use_web_search:
        web_search_result = get_web_search(query)
     else:
        web_search_result = "未使用网络搜索"

     if use_knowledge_base:
        vector_search_result = get_vector_search(query)
     else:
        vector_search_result = "未使用知识库"

     context = self_reflection(query, vector_search_result, web_search_result)

     prompt_template = """
        你是一个专业的音乐知识问答助手，名为“乐典”。你的核心职责是准确、专业、清晰地回答用户关于音乐的一切问题。
        理解与分析：仔细分析用户输入的问题，明确其核心意图和所需的知识范畴。

        信息检索与整合：
        融合信息：将内部知识、网络搜索结果以及知识库信息进行智能比对、验证与融合，形成完整的答案。

        组织与输出：你的回答应当结构清晰、重点突出、语言友好。

        用户的问题是：{question}

        与用户提问相关的信息是：{context}
        """
     prompt = ChatPromptTemplate.from_template(prompt_template)
     messages = prompt.format_messages(
                                       question=query, 
                                       context=context)
    
     llm = init_llm()
     result = llm.invoke(messages)
     return context, result.content