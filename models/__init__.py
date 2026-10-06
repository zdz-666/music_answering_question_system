"""接口层与业务层共用的 Pydantic 数据结构。

放在这里是为了避免 main（入口层）与 rag（业务层）互相 import 形成循环依赖。
"""

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    # 是否走知识库 / 网络检索由 retrieval_planner 依据提问自动决定，不再由调用方指定
    file_content: Optional[str] = None
    # 用户身份：情景记忆与用户画像按它隔离（"跨会话"= 同一 user_id 的多个 session）。
    # 缺省为空，此时整个记忆层跳过，保证评测脚本与裸调用不受影响。
    user_id: Optional[str] = None


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
    # 已把较早轮次压缩成摘要，加载历史时前端可以一并展示，避免"历史忽然变短"
    summary: Optional[str] = None


class UploadResponse(BaseModel):
    """上传接口的返回。

    入队成功（有 Redis）：`status="queued"` + `task_id`，前端轮询 /api/task/{id} 拿答案；
    Redis 不可用退回同步处理：`status="done"`，答案直接带在响应里。
    两条路返回同一个结构，前端按 status 分支即可。
    """

    status: str
    task_id: Optional[str] = None
    answer: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """异步任务状态：status ∈ queued / running / done / failed。"""

    task_id: str
    status: str
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    # 下面几项只在对应状态下出现：done 带答案，failed 带 error
    answer: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None