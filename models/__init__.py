"""接口层与业务层共用的 Pydantic 数据结构。

放在这里是为了避免 main（入口层）与 rag（业务层）互相 import 形成循环依赖。
"""

from typing import List, Optional

from pydantic import BaseModel


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