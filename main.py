import rag
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
from models import QueryRequest, ChatMessage, ChatResponse, ChatHistoryResponse
from observability import (
    new_request_id,
    set_request_id,
    reset_request_id,
    get_request_id,
    log_step,
    logger,
)
from session_store import (
    add_message,
    clear_messages,
    get_history_str,
    get_or_create_session,
    init_db,
    list_messages,
)
import zipfile
import io
import re


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 会话库建表并切到 WAL。每个 worker 启动时各跑一次，幂等。
    init_db()
    yield


app = FastAPI(
    title="Chatbot API",
    description="基于Langchain的智能问答系统API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request, call_next):
    """每个 HTTP 请求生成一个 request_id，贯穿整条 RAG 链路；结束时输出汇总。"""
    token = set_request_id(new_request_id())
    try:
        with log_step("request", path=request.url.path, method=request.method):
            response = await call_next(request)
            logger.info(
                "request.status",
                extra={"fields": {"status_code": response.status_code}},
            )
        response.headers["X-Request-Id"] = get_request_id()
        return response
    finally:
        reset_request_id(token)

# 会话存储已外置到 SQLite（session_store.py）：跨 worker 共享、进程重启不丢。
@app.post("/api/chat", response_model=ChatResponse)
async def chat(query: QueryRequest):

    session_id = get_or_create_session(query.session_id)
    history_string = get_history_str(session_id)

    # rag.get_result 是同步阻塞函数（内部是同步 LLM / requests / Chroma 调用），
    # 直接在 async 端点里调用会占住事件循环，同一 worker 上的其它请求全部排队。
    # 下沉到线程池后本进程才能并发处理请求；request_id 与 token 计量靠 contextvars 传递。
    with log_step("rag.total", input=query.question, session_id=session_id) as span:
        result = await run_in_threadpool(rag.get_result, query, history_string)
        span.output = result.content

    add_message(session_id, ChatMessage(role="user", content=query.question))
    add_message(session_id, ChatMessage(role="assistant", content=result.content))
    
    response = ChatResponse(
        answer=result.content,
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )

    return response

@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_history(session_id: str, number: int = Query(20, ge=1, le = 100)):
    messages = list_messages(session_id, number)
    if messages is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    response = ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
        timestamp=datetime.now().isoformat()
    )

    return response

@app.delete("/api/chat/history/{session_id}")
async def delete_history(session_id: str):
    if not clear_messages(session_id):
        raise HTTPException(status_code=404, detail="对话不存在")

    return {"message": "对话历史已删除",
            "session_id": session_id}

@app.post("/api/upload", response_model=ChatResponse)
async def upload_file(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    try:
        content = await file.read()
        zip_file = io.BytesIO(content)

        all_text = []

        with zipfile.ZipFile(zip_file, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('.xml'):
                    content = z.read(filename).decode('utf-8', errors='ignore')
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    all_text.append(text)
        
        file_content = " ".join(all_text)

        query_request = QueryRequest(
            question=question or "请分析上传的文件",
            session_id=session_id,
            file_content=file_content
        )
        
        session_id = get_or_create_session(query_request.session_id)
        history_string = get_history_str(session_id)
        
        with log_step(
            "rag.total",
            input={"question": query_request.question, "file": file.filename},
            session_id=session_id,
        ) as span:
            result = await run_in_threadpool(rag.get_result, query_request, history_string)
            span.output = result.content
        
        add_message(session_id, ChatMessage(role="user", content=f"已上传文件: {file.filename}" + (f"\n问题: {question}" if question else "")))
        add_message(session_id, ChatMessage(role="assistant", content=result.content))
        
        response = ChatResponse(
            answer=result.content,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

# 三个写入端点同样会阻塞：chunk_document 与 add_documents 内部要调 Embedding（网络往返），
# 语料大时是秒级，同步执行会占住事件循环，因此一并下沉到线程池。
@app.post("/api/knowledge/self-introduction")
async def add_self_introduction(text: str = Form(...)):
    try:
        await run_in_threadpool(rag.add_self_introduction, text)
        return {"message": "个人信息添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")

@app.post("/api/knowledge/music-analysis")
async def add_music_analysis(text: str = Form(...)):
    try:
        await run_in_threadpool(rag.add_music_analysis, text)
        return {"message": "音乐理解添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")

@app.post("/api/knowledge/music-list")
async def add_music_list(text: str = Form(...)):
    try:
        await run_in_threadpool(rag.add_music_list, text)
        return {"message": "歌单添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="localhost",
        port=8000,
        reload=True,  
        log_level="info"
    )