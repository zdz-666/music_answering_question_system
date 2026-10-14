import auth
import document_loader
import rag
import ratelimit
import tasks
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
from models import (
    QueryRequest,
    AuthRequest,
    AuthResponse,
    ChatMessage,
    ChatResponse,
    ChatHistoryResponse,
    TaskStatusResponse,
    UploadResponse,
)
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
    get_or_create_session,
    get_summary,
    init_db,
    list_messages,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动自检：确认会话存储依赖的 Redis 可访问。每个 worker 启动时各跑一次，幂等。
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

# 账号体系（auth.py）：把「谁在用」从随机 user_id 变成可复用的账号。
# 两个端点都不发 token——本次只做身份识别：登录成功后把该账号的 user_id 交给前端，
# 后续请求照旧带 user_id（与改造前一致）。要做访问控制得再引入 token 并在每个端点上校验。
@app.post("/api/auth/register", response_model=AuthResponse, status_code=201)
async def register(payload: AuthRequest):
    """注册新账号，成功返回它的 user_id；账号已存在返回 409。"""
    try:
        # pbkdf2 是 CPU 密集的同步计算，和 Redis 一起下沉线程池，别占住事件循环
        return await run_in_threadpool(auth.register, payload.username, payload.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(payload: AuthRequest):
    """用账号密码登录，成功返回该账号的 user_id；账号或密码不对一律 401。"""
    try:
        return await run_in_threadpool(auth.authenticate, payload.username, payload.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

# 会话存储以 Redis 为准（session_store.py）：跨 worker 共享、进程重启不丢。
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, query: QueryRequest):

    user_id = (query.user_id or "").strip()

    # 限流放在最前面：一次 chat 要跑多次 LLM 加若干外部服务，被刷的代价最高。
    # Redis 不可用时 ratelimit 内部直接放行，不会因为限流组件故障拒掉正常请求。
    await ratelimit.enforce("chat", request, user_id)

    # 一个账号只有一个会话：id 由 user_id 确定性派生，同一账号永远落在同一个会话上。
    session_id = await run_in_threadpool(get_or_create_session, user_id)

    # rag.get_result 内部走 agent_loop（同步阻塞：LLM / 检索 / 网络 / Redis），
    # 直接在 async 端点里调用会占住事件循环，同一 worker 上的其它请求全部排队。
    # 历史加载、记忆注入、工具检索与上下文压缩都在 agent 内部完成；
    # request_id 与 token 计量靠 contextvars 传递。
    with log_step("rag.total", input=query.question, session_id=session_id) as span:
        answer = await run_in_threadpool(rag.get_result, query, session_id)
        span.output = answer

    # 本轮的问答在 agent 返回后再落库：agent 是在「本轮提问落库之前」加载的历史，
    # 这样当前提问不会被重复计入上下文。Redis 写入同样下沉线程池。
    await run_in_threadpool(
        add_message, session_id, ChatMessage(role="user", content=query.question)
    )
    await run_in_threadpool(
        add_message, session_id, ChatMessage(role="assistant", content=answer)
    )

    response = ChatResponse(
        answer=answer,
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )

    return response

# 一个账号只有一个会话：历史按 user_id 查，不再需要 session_id。
@app.get("/api/chat/history", response_model=ChatHistoryResponse)
async def get_history(
    user_id: str = Query(..., min_length=1, description="用户身份，每个账号唯一一个会话"),
    number: int = Query(20, ge=1, le=100),
):
    """加载该账号唯一会话的最近消息，供聊天区回填历史。"""
    session_id = await run_in_threadpool(get_or_create_session, user_id)
    messages = await run_in_threadpool(list_messages, session_id, number) or []
    summary = await run_in_threadpool(get_summary, session_id)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
        timestamp=datetime.now().isoformat(),
        summary=summary,
    )

@app.post("/api/upload", response_model=UploadResponse, status_code=202)
async def upload_file(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    # 限流放在 file.read() 之前：上传要走 VLM 逐张识图，比 chat 更贵，
    # 被反复上传的请求应该在读进内存、调模型之前就被拦下。
    await ratelimit.enforce("upload", request, user_id)

    content = await file.read()
    user_id = (user_id or "").strip()

    # 有 Redis 就入队，立刻返回 task_id（HTTP 202），解析与生成交给 worker.py；
    # 没配 REDIS_URL / Redis 连不上时退回同步处理（status=done，答案直接带回来）。
    # 队列是可选设施，不能让「没装 Redis」变成「上传不可用」——与缓存层同一条原则。
    # 写暂存文件是阻塞 IO，一并下沉线程池。
    task_id = await run_in_threadpool(
        tasks.submit,
        filename=file.filename,
        content=content,
        question=question,
        user_id=user_id,
        request_id=get_request_id(),
    )
    if task_id:
        return UploadResponse(status="queued", task_id=task_id)

    response.status_code = 200
    try:
        result = await run_in_threadpool(
            rag.answer_with_file,
            filename=file.filename,
            content=content,
            question=question,
            user_id=user_id,
        )
    except document_loader.UnsupportedFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(exc)}")

    return UploadResponse(status="done", **result)


@app.get("/api/task/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    """轮询异步任务状态：queued / running / done / failed。

    任务记录存在 Redis 里（TTL 由 TASK_TTL_SECONDS 控制），过期后这里返回 404。
    Redis 连不上时同样查不到 —— 此时 /api/upload 会走同步处理，本来也不产生任务。
    """
    record = await run_in_threadpool(tasks.load, task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    return TaskStatusResponse(
        task_id=task_id,
        status=record.get("status", "queued"),
        created_at=record.get("created_at"),
        finished_at=record.get("finished_at"),
        answer=record.get("answer"),
        session_id=record.get("session_id"),
        timestamp=record.get("timestamp"),
        error=record.get("error"),
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