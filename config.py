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

# Chroma 持久化目录：知识库（三个集合）的数据落在这里，已在 .gitignore 中排除。
# 会话在 Redis、长期记忆在 MEMORY_ROOT 下的文件里，都不占用这个目录。
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

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

# ---- 长期记忆（文件式，按 user_id 隔离）----
# 记忆不再走向量检索，而是像 memory_example 那样落成一个个 Markdown 文件：
# 一个 user_id 一个目录（MEMORY_ROOT/<safe_dir(user_id)>/），目录里每条记忆一个 .md，
# 外加一份 MEMORY.md 目录索引。回答前用 LLM 从目录索引里挑出相关记忆拼进 system prompt，
# 一轮结束后再抽取新的持久记忆落盘。详见 memory/ 包（store / recall / learn）。
#
# 记忆根目录：按 user_id 分目录存放，已在 .gitignore 中排除。
MEMORY_ROOT = os.getenv("MEMORY_ROOT", "./.memory")
# 单次注入 system prompt 的记忆正文总字符上限（recall 时按此截断）
MEMORY_RECALL_CHAR_LIMIT = int(os.getenv("MEMORY_RECALL_CHAR_LIMIT", "20000"))
# 单次最多召回几条记忆
MEMORY_RECALL_TOP_K = int(os.getenv("MEMORY_RECALL_TOP_K", "5"))
# 记忆条数达到该阈值时触发一次合并（consolidate），把重复/过时的记录整理掉
MEMORY_CONSOLIDATE_THRESHOLD = int(os.getenv("MEMORY_CONSOLIDATE_THRESHOLD", "10"))
# 合并时送入模型的目录正文上限；超过则跳过本次合并
MEMORY_CONSOLIDATE_CHAR_LIMIT = int(
    os.getenv("MEMORY_CONSOLIDATE_CHAR_LIMIT", "20000")
)
# read_file 工具（agent 读取记忆文件与压缩归档）单次返回的字符上限，
# 避免把大文件整段灌进上下文；白名单目录见 file_access.py。
READ_FILE_CHAR_LIMIT = int(os.getenv("READ_FILE_CHAR_LIMIT", "20000"))

# ---- 上下文压缩（agent_loop 每轮调用前对聊天消息做瘦身）----
# 阈值含义与 context_compact_example.py 一致，见 context_compact.py：
#   CONTEXT_CHAR_LIMIT       消息总量超过它就逐级压缩
#   TOOL_RESULT_BATCH_CHAR_LIMIT  单批工具结果超过它就把超大结果落盘
#   LARGE_RESULT_CHAR_LIMIT  单条工具结果的落盘门槛
#   SUMMARY_INPUT_CHAR_LIMIT 汇总历史时送入模型的正文上限
#   KEEP_RECENT_RESULTS      微压缩时保留最近几条工具结果原文
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "50000"))
TOOL_RESULT_BATCH_CHAR_LIMIT = int(os.getenv("TOOL_RESULT_BATCH_CHAR_LIMIT", "200000"))
LARGE_RESULT_CHAR_LIMIT = int(os.getenv("LARGE_RESULT_CHAR_LIMIT", "30000"))
SUMMARY_INPUT_CHAR_LIMIT = int(os.getenv("SUMMARY_INPUT_CHAR_LIMIT", "80000"))
KEEP_RECENT_RESULTS = int(os.getenv("KEEP_RECENT_RESULTS", "3"))
KEEP_RECENT_MESSAGES = int(os.getenv("KEEP_RECENT_MESSAGES", "5"))
# snip 压缩的消息条数上限：超过就把中段归档成一份 transcript
CONVERSATION_SNIP_LIMIT = int(os.getenv("CONVERSATION_SNIP_LIMIT", "50"))
# 归档 transcript 与落盘工具结果的目录，已在 .gitignore 中排除。
# 运行时会在其下按 user_id 再分一层子目录（与记忆同一套 safe_dir 命名），见 context_compact.py。
TRANSCRIPT_DIR = os.getenv("TRANSCRIPT_DIR", "./.transcripts")
TOOL_RESULTS_DIR = os.getenv("TOOL_RESULTS_DIR", "./.task_outputs/tool-results")
# 每次请求从 Redis 会话里加载的历史消息条数上限，作为 agent 的初始上下文。
# 默认与 MAX_MESSAGES_HARD 对齐 = 一次把会话里现存的消息全加载进来：
# 压缩提交会按「已加载消息的最大 id」删除被摘要覆盖的那批，若这里截断就会删到
# 没进摘要的旧消息，因此不要调得比 MAX_MESSAGES_HARD 小。
AGENT_HISTORY_LIMIT = int(os.getenv("AGENT_HISTORY_LIMIT", "200"))
# 返回边界上的答案验收（agent._judge_answer）：连续判定「未完成」的拦截上限，
# 超过就带着当前答案收尾；评估器自身报错时也是直接收尾（fail-open），不空转。
ANSWER_BLOCK_CAP = int(os.getenv("ANSWER_BLOCK_CAP", "3"))
# 子智能体（父智能体的 task 工具）自己那条 agent loop 的轮数上限
SUBAGENT_MAX_ROUNDS = int(os.getenv("SUBAGENT_MAX_ROUNDS", "10"))

# ---- 账号（auth.py：用户名 + 密码换 user_id）----
# pbkdf2-hmac-sha256 的迭代次数。调高更抗暴力破解、但每次登录/注册都更慢；
# 默认 10 万次在本机约几十毫秒，登录体验与强度都够用。
AUTH_PBKDF2_ITERATIONS = int(os.getenv("AUTH_PBKDF2_ITERATIONS", "100000"))

# ---- Redis（会话 / 缓存 / 限流 / 任务队列）----
# 定位分两档，别混为一谈：
#   必需 —— 会话（session_store）：拿不到连接就抛错，不做降级，
#            因为它承载的是数据本身（读不到历史会伪装成「新对话」的静默数据错误）；
#   可选 —— 缓存 / 限流 / 上传队列：拿不到就静默退化（每次现算、限流放行、上传同步处理）。
# 长期记忆已改为文件式（MEMORY_ROOT），不再占用 Redis。
# 本地起实例的命令见 README。
REDIS_URL = os.getenv("REDIS_URL", "")
# 所有键的版本前缀。改这里的版本号（v1 → v2）即可让旧格式缓存整体失联，
# 不必去翻一遍键；知识库写入则用集合级版本号做定向失效。
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "musicrag:v1")
# 会话相关键（meta / msgs / sum）的存活时间（秒）。每写一次消息就续期一轮，
# 即「最后一次活动之后 30 天过期」；过期的会话查历史会得到 404。
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "2592000"))
# 各层缓存的存活时间（秒）。越靠前的层输入越稳定、越靠后的层越容易过期：
# 嵌入只由文本决定，给一天；集合路由是 LLM 的确定性映射，给一小时；
# 检索结果绑定集合内容，给一小时；最终答案受联网结果影响，只给几分钟。
CACHE_TTL_EMBEDDING = int(os.getenv("CACHE_TTL_EMBEDDING", "86400"))
CACHE_TTL_LLM = int(os.getenv("CACHE_TTL_LLM", "3600"))
CACHE_TTL_RETRIEVAL = int(os.getenv("CACHE_TTL_RETRIEVAL", "3600"))
CACHE_TTL_ANSWER = int(os.getenv("CACHE_TTL_ANSWER", "300"))

# ---- 接口限流（令牌桶，需要 Redis；未配 REDIS_URL 时不生效）----
# 开关：置 false 可整体关掉限流（压测、演示时用）
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() not in ("0", "false", "no")
# 容量 = 允许的突发次数；补充速率 = 长期平均速率（次/秒）
# /api/chat 一次要跑 6 次 LLM + 若干外部服务，给宽松些：
# 平时 0.2/s（约 12 次/分钟），最多突发 20 次
RATE_LIMIT_CHAT_CAPACITY = float(os.getenv("RATE_LIMIT_CHAT_CAPACITY", "20"))
RATE_LIMIT_CHAT_REFILL_PER_SEC = float(os.getenv("RATE_LIMIT_CHAT_REFILL_PER_SEC", "0.2"))
# /api/upload 要走 VLM 逐张识图，单次成本远高于 chat，配额收紧：
# 平时 0.05/s（约 3 次/分钟），最多突发 5 次
RATE_LIMIT_UPLOAD_CAPACITY = float(os.getenv("RATE_LIMIT_UPLOAD_CAPACITY", "5"))
RATE_LIMIT_UPLOAD_REFILL_PER_SEC = float(
    os.getenv("RATE_LIMIT_UPLOAD_REFILL_PER_SEC", "0.05")
)

# ---- 上传任务队列（需要 Redis；未配 REDIS_URL 时上传退回同步处理）----
# 上传文档要逐张调 VLM 识图，几十张图的 PDF 能跑几分钟，同步等会把 HTTP 连接一直挂住。
# 有 Redis 时改成「入队 + 前端轮询」：任务记录与结果的存活时间（秒），够轮询拿到即可。
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "7200"))
# 排队期间上传文件的暂存目录：Redis 里只存路径，不塞几十 MB 的字节流。
# 已在 .gitignore 中排除，worker 处理完会立刻删掉对应文件。
UPLOAD_SPOOL_DIR = os.getenv("UPLOAD_SPOOL_DIR", "./uploaded_files")


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
    """向量模型实例，供分块与 Chroma 集合共用，必须与建库时保持一致。

    外面包一层 Redis 查询缓存：检索时每次都要把查询文本送去做一次嵌入，
    而同一个查询在一次请求里会被多集合轮询、补充检索反复用到。
    这么包而不是在各个调用点手动缓存，是因为 data_storage / dynamic_chunk /
    memory 都从这里取向量模型，包在这里能一次覆盖全部调用方。
    """
    # 函数内 import：cache 需要读本模块的 REDIS_URL 等常量，模块级 import 会成环
    from cache import CachedEmbeddings

    return CachedEmbeddings(
        OpenAIEmbeddings(
            model=_env("EMBEDDING_MODEL"),
            api_key=_env("OPENAI_API_KEY"),
            base_url=_env("OPENAI_BASE_URL"),
        )
    )


def get_tavily_api_key() -> str:
    return _env("TAVILY_API_KEY")


def get_rerank_api_key() -> str:
    return _env("RERANK_API_KEY")