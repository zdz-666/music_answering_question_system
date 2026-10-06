from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

import cache
from config import CHROMA_PERSIST_DIR, get_embeddings
from dynamic_chunk import SemanticChunker
from observability import logger

# 混合检索的融合权重（稠密 : 稀疏）。
# ChromaDB 没有内置 BM25，这里用 EnsembleRetriever 的加权 RRF 复现原先
# Milvus weighted ranker 的 0.6/0.4 语义。
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4


def get_client() -> PersistentClient:
    """Chroma 持久化客户端，数据落在 CHROMA_PERSIST_DIR 目录。"""
    return PersistentClient(path=CHROMA_PERSIST_DIR)


def list_collections() -> set:
    """已有集合名集合。兼容 chromadb 不同版本返回名称或 Collection 对象的差异。"""
    return {
        c if isinstance(c, str) else c.name
        for c in get_client().list_collections()
    }


def collection_create(url, collection_name):
    embeddings = get_embeddings()
    text_splitter = SemanticChunker(
            overlap_size=0,
            max_chunk_size=300
        )

    loader = TextLoader(url, encoding='utf-8')
    docs = loader.load()
    docs_text = docs[0].page_content

    split_docs = text_splitter.chunk_document(docs_text)

    vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR,
)

    # 集合内容变了，该集合的检索缓存必须立刻作废（缓存的键里带着这个版本号）
    cache.bump_kb_version(collection_name)

    if collection_name in list_collections():
        logger.info("collection.create", extra={"fields": {"collection": collection_name, "result": "success", "docs": len(split_docs)}})
    else:
        logger.error("collection.create", extra={"fields": {"collection": collection_name, "result": "failed"}})

    return vector_store


def load_vector_store(collection_name: str) -> Chroma:
    """打开已有集合，新增文档与检索共用。集合不存在时 Chroma 会自动建空集合。"""
    vector_store_loaded = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store_loaded


def load_all_documents(collection_name: str) -> list:
    """取出集合内全部文档，用于构建 BM25 稀疏索引（Chroma 本身只存稠密向量）。"""
    data = get_client().get_collection(collection_name).get(
        include=["documents", "metadatas"]
    )
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(
            data.get("documents") or [],
            data.get("metadatas") or [],
        )
    ]


def vector_similarity_search(collection_name, query, k):
    """稠密 + 稀疏混合检索，返回按融合分数排序的 List[Document]。

    替代原先 Milvus 的 dense/sparse 双向量字段与 weighted ranker：
    稠密部分走 Chroma 向量检索，稀疏部分走 BM25，再按 DENSE_WEIGHT/SPARSE_WEIGHT 融合。
    """
    dense_retriever = load_vector_store(collection_name).as_retriever(
        search_kwargs={"k": k}
    )

    documents = load_all_documents(collection_name)
    if not documents:
        return []

    sparse_retriever = BM25Retriever.from_documents(documents, k=k)

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return ensemble.invoke(query)


def in_memory_similarity_search(documents: list, query, k):
    """在调用方给的一批文本上做「稠密 + 稀疏」混合检索，语义与 vector_similarity_search 一致。

    给语料不在 Chroma 里的场景用（评测时把外部语料当作知识库）：向量索引只在本次调用内
    构建，跑完即丢，既不落盘也不影响已有集合。
    """
    docs = [Document(page_content=text) for text in documents if str(text).strip()]
    if not docs:
        return []

    # 语料可能比 k 还小（评测时一条样本只有几篇文档），先收敛 k，避免底层越界
    k = min(k, len(docs))

    dense_retriever = InMemoryVectorStore.from_documents(
        docs, embedding=get_embeddings()
    ).as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(docs, k=k)

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return ensemble.invoke(query)


def drop_collection(collection_name):
    get_client().delete_collection(collection_name)
    # 集合没了，它的检索缓存也不能再被命中
    cache.bump_kb_version(collection_name)

    if collection_name in list_collections():
        logger.error("collection.drop", extra={"fields": {"collection": collection_name, "result": "failed"}})
    else:
        logger.info("collection.drop", extra={"fields": {"collection": collection_name, "result": "success"}})

    return None