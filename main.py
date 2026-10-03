import document_loader
import memory
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
    get_summary,
    init_db,
    list_messages,
)


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

    user_id = (query.user_id or "").strip()
    session_id = get_or_create_session(query.session_id)
    history_string = get_history_str(session_id)

    # 情景记忆 / 用户画像的相似度检索要走 Embedding 网络往返，必须下沉线程池
    with log_step(
        "memory.build_context", session_id=session_id, user_id=user_id
    ) as memory_span:
        memory_string = await run_in_threadpool(
            memory.build_context, user_id, session_id, query.question
        )
        memory_span.output = memory_string

    # rag.get_result 是同步阻塞函数（内部是同步 LLM / requests / Chroma 调用），
    # 直接在 async 端点里调用会占住事件循环，同一 worker 上的其它请求全部排队。
    # 下沉到线程池后本进程才能并发处理请求；request_id 与 token 计量靠 contextvars 传递。
    with log_step("rag.total", input=query.question, session_id=session_id) as span:
        result = await run_in_threadpool(
            rag.get_result, query, history_string, memory_string
        )
        span.output = result.content

    add_message(session_id, ChatMessage(role="user", content=query.question))
    add_message(session_id, ChatMessage(role="assistant", content=result.content))

    # 一轮结束后更新记忆：压缩工作记忆（LLM）+ 写情景记忆/画像（Embedding），同样是阻塞操作
    await run_in_threadpool(
        memory.after_turn, user_id, session_id, query.question, result.content
    )

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
        timestamp=datetime.now().isoformat(),
        summary=get_summary(session_id),
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
    user_id: Optional[str] = Form(None),
):
    content = await file.read()

    # 解析在 document_loader 里分发：PDF（PyMuPDF）/ DOCX（python-docx）/ 独立图片，
    # 文档内嵌图与独立图片都会交给 VLM 转成中文描述，一起拼进 file_content。
    # 读文件与逐张调 VLM 都是同步阻塞的，必须下沉线程池。
    with log_step(
        "document.parse",
        input={"file": file.filename, "bytes": len(content)},
    ) as parse_span:
        try:
            blocks = await run_in_threadpool(
                document_loader.parse_document, file.filename, content
            )
        except document_loader.UnsupportedFormatError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"文件处理失败: {str(exc)}")

        file_content = document_loader.blocks_to_text(blocks)
        parse_span.output = {
            "blocks": len(blocks),
            "images": sum(1 for block in blocks if block.kind == "image"),
            "chars": len(file_content),
        }

    query_request = QueryRequest(
        question=question or "请分析上传的文件",
        session_id=session_id,
        file_content=file_content
    )

    user_id = (user_id or "").strip()
    session_id = get_or_create_session(query_request.session_id)
    history_string = get_history_str(session_id)

    # 与 /api/chat 同构：先取三层记忆，再交给检索与生成；上传场景的记忆检索同样要下沉线程池
    with log_step(
        "memory.build_context", session_id=session_id, user_id=user_id
    ) as memory_span:
        memory_string = await run_in_threadpool(
            memory.build_context, user_id, session_id, query_request.question
        )
        memory_span.output = memory_string

    with log_step(
        "rag.total",
        input={"question": query_request.question, "file": file.filename},
        session_id=session_id,
    ) as span:
        result = await run_in_threadpool(
            rag.get_result, query_request, history_string, memory_string
        )
        span.output = result.content

    add_message(session_id, ChatMessage(role="user", content=f"已上传文件: {file.filename}" + (f"\n问题: {question}" if question else "")))
    add_message(session_id, ChatMessage(role="assistant", content=result.content))

    await run_in_threadpool(
        memory.after_turn, user_id, session_id, query_request.question, result.content
    )

    return ChatResponse(
        answer=result.content,
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )

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