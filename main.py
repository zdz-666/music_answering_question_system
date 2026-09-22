import rag
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import base64
import zipfile
import io
import re

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_web_search: bool = True
    use_knowledge_base: bool = True
    file_content: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total: int
    timestamp: str


app = FastAPI(
    title="Chatbot API",
    description="基于Langchain的智能问答系统API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

"""
sessions{str : {"created_at": str, "messages": List[ChatMessage]}}
"""
#获取或创建对话
def get_session(session_id: str):
    if session_id and session_id in sessions:
        return session_id
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(sessions)}"
    sessions[new_session_id] = {
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    return new_session_id

#添加消息到会话
def add_message(session_id: str, message: ChatMessage):
    if session_id in sessions:
        message = ChatMessage(
            role=message.role,
            content=message.content,
            timestamp=datetime.now().isoformat()
        )
        sessions[session_id]["messages"].append(message)

        # 限制历史消息为20条
        if len(sessions[session_id]["messages"]) > 20:
            sessions[session_id]["messages"] = sessions[session_id]["messages"][-20:]

def get_history_str(session_id: str):
    messages = sessions[session_id]["messages"][-4:]
    history_string = ""
    for msg in messages:
        if msg.role == "user":
            history_string += f"用户: {msg.content}\n"
        elif msg.role == "assistant":
            history_string += f"助手: {msg.content}\n"
    return history_string
@app.post("/api/chat", response_model=ChatResponse)
async def chat(query: QueryRequest):

    session_id = get_session(query.session_id)
    history_string = get_history_str(session_id)

    result = rag.get_result(query, history_string)

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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    messages = sessions[session_id]["messages"][-number:]
    response = ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
        timestamp=datetime.now().isoformat()
    )

    return response

@app.delete("/api/chat/history/{session_id}")
async def delete_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    sessions[session_id]["messages"] = []
    return {"message": "对话历史已删除",
            "session_id": session_id}

@app.post("/api/upload", response_model=ChatResponse)
async def upload_file(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    use_web_search: bool = Form(True),
    use_knowledge_base: bool = Form(True)
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
            use_web_search=use_web_search,
            use_knowledge_base=use_knowledge_base,
            file_content=file_content
        )
        
        session_id = get_session(query_request.session_id)
        history_string = get_history_str(session_id)
        
        result = rag.get_result(query_request, history_string)
        
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

@app.post("/api/knowledge/self-introduction")
async def add_self_introduction(text: str = Form(...)):
    try:
        rag.add_self_introduction(text)
        return {"message": "个人信息添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")

@app.post("/api/knowledge/music-analysis")
async def add_music_analysis(text: str = Form(...)):
    try:
        rag.add_music_analysis(text)
        return {"message": "音乐理解添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")

@app.post("/api/knowledge/music-list")
async def add_music_list(text: str = Form(...)):
    try:
        rag.add_music_list(text)
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