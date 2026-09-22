from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from dynamic_chunk import SemanticChunker
from pymilvus import connections,utility
from langchain_milvus import Milvus, BM25BuiltInFunction

def collection_create(url, collection_name):
    embeddings = OpenAIEmbeddings(
            model="",
            api_key="",
            base_url=""
)
    text_splitter = SemanticChunker(
            overlap_size=0,
            max_chunk_size=300
        )
    
    loader = TextLoader(url, encoding='utf-8')
    docs = loader.load()
    docs_text = docs[0].page_content

    split_docs = text_splitter.chunk_document(docs_text)

    conn = connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
    vector_store = Milvus.from_documents(
            documents=split_docs,
            embedding=embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            collection_name=collection_name,
            connection_args={
                "host": "localhost",
                "port": "19530",
            },
            drop_old=False,  
            auto_id=True,
            consistency_level="Strong",
)
    
    if utility.has_collection(collection_name):
        print(f"{collection_name} 创建成功")
    else:
        print(f"{collection_name} 创建失败")

    return vector_store


def milvus_similarity_search(collection_name, query, k):
    embeddings = OpenAIEmbeddings(
            model="",
            api_key="",
            base_url=""
            )
    conn = connections.connect(
    alias="default",
    host="localhost",
    port="19530"
    )
    vector_store = Milvus(
    embedding_function=embeddings,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    collection_name=collection_name,
    connection_args={
        "host": "localhost",
        "port": "19530",
    },
)
    result = vector_store.similarity_search_with_score(query, k=k, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})

    return result


def drop_collection(collection_name):
    conn = connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
    utility.drop_collection(collection_name)

    if utility.has_collection(collection_name):
        print(f"{collection_name} 删除失败")
    else:
        print(f"{collection_name} 删除成功")

    return None

def create_vector_store_loaded(coll_name: str):
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    embeddings = OpenAIEmbeddings(
            model="",
            api_key="",
            base_url=""
            )
    vector_store_loaded = Milvus(
    embedding_function=embeddings,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    connection_args={
                "host": "localhost",
                "port": "19530",
            },
    collection_name=coll_name,
)
    return vector_store_loaded