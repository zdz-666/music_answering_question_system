"""模型、密钥与向量库路径的统一配置入口。

模型名 / api_key / base_url / 连接地址原先散落在 rag、collection_router、
data_storage、dynamic_chunk、llm_ev 五个文件里，改一处要动五处；
现在全部收敛到本文件，调用方只 import 这里的工厂函数。

所有敏感值通过 .env 读取（.env 已被 .gitignore 排除，不会进入 git）。
本文件只保存变量名与安全的默认值，不保存任何真实密钥。
配置方式和变量清单见 .env.example。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from observability import TokenUsageCallback

load_dotenv()


def _env(name: str, default: str | None = None) -> str:
    """读取环境变量；缺失时抛出带操作提示的错误，而不是让下游报出难懂的 401。"""
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}：请复制 .env.example 为 .env 并填写后重试。"
        )
    return value


# 只有具备通用默认值的项才在这里取默认，密钥与模型名一律要求显式配置
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1/rerank")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")

# ---- 外部服务的超时与重试 ----
# 超时是必须的：原先 requests.post 不带 timeout，上游挂住会一直占着工作线程
# 单次 10s 对重排/搜索足够（正常 1~3s），最坏情况 = 超时 x (HTTP_RETRIES + 1)
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "10"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
# 重试次数（不含首次），即最多尝试 HTTP_RETRIES + 1 次
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "2"))

# ---- 检索策略 ----
# 首轮检索之后最多追加几轮补充检索（硬上限，防止无限循环）；0 表示关闭补充检索
MAX_SUPPLEMENT_ROUNDS = int(os.getenv("MAX_SUPPLEMENT_ROUNDS", "2"))

# Chroma 持久化目录：向量库数据落在这里，已在 .gitignore 中排除
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 会话库：SQLite 文件路径，同样已在 .gitignore 中排除
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "./sessions.db")

# ---- 对话模型成本（可选，用于日志里的 cost_usd 估算）----
# 未配置时为 0，此时只记录 token 数，不折算金额
LLM_INPUT_PRICE_PER_MT = float(os.getenv("LLM_INPUT_PRICE_PER_MT", "0"))
LLM_OUTPUT_PRICE_PER_MT = float(os.getenv("LLM_OUTPUT_PRICE_PER_MT", "0"))

# ---- 上传文档解析（PDF / DOCX）----
# 图像理解模型（VLM）：与对话模型共用同一个 OpenAI 兼容网关，只需换 model 名。
# 留空表示不启用图片描述——上传仍可用，只是图片位置会标注为「未生成描述」。
VLM_MODEL = os.getenv("VLM_MODEL", "")
# 单张图片描述的等待上限；图片比文本慢，默认给到 30s
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "30"))
# 宽或高小于该像素的图片直接丢弃（滤掉 logo、页码装饰、公式小图标）
IMAGE_MIN_SIZE = int(os.getenv("IMAGE_MIN_SIZE", "100"))
# 抽取出的图片与描述缓存落盘目录，已在 .gitignore 中排除
UPLOAD_IMAGE_DIR = os.getenv("UPLOAD_IMAGE_DIR", "./uploaded_images")

# ---- 三层记忆（工作记忆 / 情景记忆 / 用户画像）----
# 情景记忆：跨会话的历史问答，按 user_id 过滤 + 语义相似度检索
# 用户画像：从对话里抽取的长期偏好与实体，一条一个原子事实
EPISODIC_COLLECTION = os.getenv("EPISODIC_COLLECTION", "episodic_memory")
PROFILE_COLLECTION = os.getenv("PROFILE_COLLECTION", "user_profile")
# 注入 prompt 的记忆条数上限
EPISODIC_TOP_K = int(os.getenv("EPISODIC_TOP_K", "3"))
PROFILE_TOP_K = int(os.getenv("PROFILE_TOP_K", "5"))
# 单条记忆注入 prompt 时的截断长度，防止一条长回答把上下文挤爆
MEMORY_SNIPPET_CHARS = int(os.getenv("MEMORY_SNIPPET_CHARS", "200"))

# 情景记忆的时间衰减：recency = exp(-EPISODIC_DECAY_FACTOR * age_hours / 24)
# 0.1 时：1 天前约 0.90、1 周前约 0.50、1 个月前约 0.05（越近的对话越容易被召回）
EPISODIC_DECAY_FACTOR = float(os.getenv("EPISODIC_DECAY_FACTOR", "0.1"))
# 情景记忆的最终得分 = (1 - w) * 向量相似度 + w * 时间近因性
EPISODIC_RECENCY_WEIGHT = float(os.getenv("EPISODIC_RECENCY_WEIGHT", "0.3"))


def get_chat_model(temperature: float = 0.1) -> ChatOpenAI:
    """对话模型实例。原先各文件各自 new 一个 ChatOpenAI，现在统一从这里取。"""
    return ChatOpenAI(
        model=_env("LLM_MODEL"),
        api_key=_env("OPENAI_API_KEY"),
        base_url=_env("OPENAI_BASE_URL"),
        temperature=temperature,
        callbacks=[
            TokenUsageCallback(
                input_price_per_mtok=LLM_INPUT_PRICE_PER_MT,
                output_price_per_mtok=LLM_OUTPUT_PRICE_PER_MT,
            )
        ],
    )


def get_vlm(temperature: float = 0.2) -> ChatOpenAI:
    """图像理解模型（VLM）实例：与对话模型共用网关，仅 model 名不同。

    超时与重试直接用 ChatOpenAI 自带的 timeout / max_retries，不再套 tenacity：
    图片描述失败已在 document_loader 里降级为占位文本，不影响整篇解析。
    """
    return ChatOpenAI(
        model=_env("VLM_MODEL"),
        api_key=_env("OPENAI_API_KEY"),
        base_url=_env("OPENAI_BASE_URL"),
        temperature=temperature,
        timeout=VLM_TIMEOUT,
        max_retries=HTTP_RETRIES,
        callbacks=[
            TokenUsageCallback(
                input_price_per_mtok=LLM_INPUT_PRICE_PER_MT,
                output_price_per_mtok=LLM_OUTPUT_PRICE_PER_MT,
            )
        ],
    )


def get_embeddings() -> OpenAIEmbeddings:
    """向量模型实例，供分块与 Chroma 集合共用，必须与建库时保持一致。"""
    return OpenAIEmbeddings(
        model=_env("EMBEDDING_MODEL"),
        api_key=_env("OPENAI_API_KEY"),
        base_url=_env("OPENAI_BASE_URL"),
    )


def get_tavily_api_key() -> str:
    return _env("TAVILY_API_KEY")


def get_rerank_api_key() -> str:
    return _env("RERANK_API_KEY")