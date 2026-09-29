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

# Chroma 持久化目录：向量库数据落在这里，已在 .gitignore 中排除
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_chat_model(temperature: float = 0.1) -> ChatOpenAI:
    """对话模型实例。原先各文件各自 new 一个 ChatOpenAI，现在统一从这里取。"""
    return ChatOpenAI(
        model=_env("LLM_MODEL"),
        api_key=_env("OPENAI_API_KEY"),
        base_url=_env("OPENAI_BASE_URL"),
        temperature=temperature,
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