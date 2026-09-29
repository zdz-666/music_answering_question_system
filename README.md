# 乐典音乐助手（Music Chatbot）

一个基于 **RAG（检索增强生成）** 的音乐知识问答系统。后端使用 FastAPI + LangChain 编排
「查询重写 → 智能路由 → 向量/网络检索 → 重排序 → 自反思过滤 → 生成」的完整链路，
前端使用 React 提供聊天界面，向量数据由 ChromaDB 本地持久化。

---

## 功能特性

- **多路检索融合**：同时支持本地知识库检索与联网搜索，用户可在前端自由开关。
- **查询重写**：把口语化的用户提问改写为适合向量检索的正式查询；联网搜索前再单独做一次关键词净化。
- **智能路由（Collection Router）**：由 LLM 判断问题应该落到哪个集合，支持多集合同时命中。
- **混合检索 + 重排序**：Chroma 稠密检索与 BM25 稀疏检索按 0.6/0.4 加权融合，再用 Reranker 模型按相关性截取前 50%。
- **自反思过滤（Self-Reflection）**：在生成答案前，先由 LLM 从知识库/网络结果中剔除无关信息。
- **多轮对话**：后端按 `session_id` 维护会话，最多保留最近 20 条消息，生成时注入最近 4 条作为上下文。
- **文件问答**：上传 `.docx`（zip + XML）后解析正文，结合用户问题一起回答。
- **知识库动态写入**：可在前端直接录入「个人信息 / 音乐理解 / 歌单」三类私人数据。
- **离线评测（llm_ev.py）**：内置 BLEU、ROUGE 与 LLM 忠诚度打分（0/1/2）脚本。

---

## 技术栈

| 层次 | 技术 |
| --- | --- |
| 后端框架 | FastAPI + Uvicorn |
| LLM 编排 | LangChain（`langchain-openai`、`langchain-chroma`、`langchain-classic`、`langchain-community`） |
| 向量数据库 | ChromaDB 1.4.0（本地持久化，无需独立服务或 Docker） |
| 检索 | Chroma 稠密检索 + BM25 稀疏检索，经 `EnsembleRetriever` 加权融合，再由 Reranker（SiliconFlow `Qwen/Qwen3-Reranker-8B`）截取前 50% |
| 联网搜索 | Tavily |
| 分词/评测 | jieba、nltk（BLEU）、rouge-score |
| 前端 | React 19 + Create React App + axios |

---

## 目录结构

```
bishe/
├── main.py                 # FastAPI 入口，路由、会话管理与文件上传
├── rag.py                  # RAG 主链路：查询重写、检索、自反思、答案生成
├── collection_router.py    # LLM 结构化输出做集合路由（含独立测试入口）
├── data_storage.py         # Chroma 集合创建/删除、稠密+BM25 混合检索
├── dynamic_chunk.py        # 基于句子语义相似度的动态分块（SemanticChunker）
├── config.py               # 模型 / 密钥 / Chroma 路径的统一配置入口
├── models/                 # 接口层与业务层共用的 Pydantic 数据结构
├── .env.example            # 环境变量模板（复制为 .env 后填写）
├── llm_ev.py               # 评测脚本：BLEU / ROUGE / LLM 忠诚度
├── self_data/              # 示例私人语料（自我介绍、音乐分析、歌单）
├── chroma_db/              # Chroma 持久化数据（运行时生成，不入库）
└── music-chatbot-frontend/ # React 前端
    └── src/
        ├── App.js          # 聊天界面、会话管理、知识库录入、文件上传
        └── service/api.js  # axios 封装的后端接口调用
```

`models/` 的存在是为了避免 `main`（入口层）与 `rag`（业务层）互相 import；
所有配置读取都收敛在 `config.py`，业务文件里不再出现模型名与密钥。

---

## 工作流程

```
用户提问
   │
   ├─ 查询重写 (query_rewriting) ──► 适合向量检索的查询
   │        └─ 集合路由 (get_router_collection)
   │               └─ Chroma 稠密 + BM25 稀疏混合检索 (k=6) ──► Reranker 取前 50%
   │
   ├─ 网络搜索 (web_rewriting → Tavily)
   │
   ▼
自反思过滤 (self_reflection) ──► 与问题相关的知识库信息 + 网络信息
   │
   ▼
答案生成 (乐典助手 Prompt + 历史对话 + 上传文件)
   │
   ▼
返回答案 & 记录会话
```

分块策略见 [dynamic_chunk.py](file:///d:/bishe/dynamic_chunk.py)：先按中英文标点切句，逐句向量化，
相邻句余弦相似度低于 `0.7` 处切分，再按 `max_chunk_size` 强制截断并支持重叠。

---

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 发送消息，返回答案与 `session_id` |
| GET | `/api/chat/history/{session_id}` | 查询会话历史（`number` 可选，默认 20） |
| DELETE | `/api/chat/history/{session_id}` | 清空指定会话历史 |
| POST | `/api/upload` | 上传 `.docx` 文件并提问 |
| POST | `/api/knowledge/self-introduction` | 写入个人信息 |
| POST | `/api/knowledge/music-analysis` | 写入音乐理解 |
| POST | `/api/knowledge/music-list` | 写入歌单 |

服务启动后可在 `http://localhost:8000/docs` 查看 Swagger 文档。

---

## 快速开始

ChromaDB 以嵌入式方式运行，数据直接落盘到 `CHROMA_PERSIST_DIR`，
**不需要 Docker，也不需要单独启动数据库服务**。

### 1. 后端依赖

```bash
pip install fastapi uvicorn langchain langchain-openai langchain-chroma langchain-classic \
    langchain-community chromadb rank-bm25 tavily-python requests jieba nltk rouge-score \
    numpy torch python-multipart python-dotenv
```

### 2. 配置模型与密钥

模型名、`api_key`、`base_url`、Chroma 存储路径全部集中在 [config.py](file:///d:/bishe/config.py)，
通过环境变量读取，不再散落在各业务文件里：

```bash
cp .env.example .env    # 然后填写 .env
```

`.env` 已被 [.gitignore](file:///d:/bishe/.gitignore) 排除，密钥不会进入 git。
[config.py](file:///d:/bishe/config.py) 只保存变量名与安全的默认值，不含任何真实密钥。

需要填写的变量见 [.env.example](file:///d:/bishe/.env.example)：`OPENAI_API_KEY` / `OPENAI_BASE_URL` /
`LLM_MODEL` / `EMBEDDING_MODEL`（对话与向量模型，OpenAI 兼容接口）、`TAVILY_API_KEY`（联网搜索）、
`RERANK_API_KEY`（重排序）、`CHROMA_PERSIST_DIR`（向量库落盘目录，默认 `./chroma_db`）。

业务侧统一通过 `config.get_chat_model()`、`config.get_embeddings()` 取实例，
因此换模型只需改一处。缺失变量会抛出带操作提示的 `RuntimeError`，
而不是在调用 API 时才报出难懂的 401。

### 3. 初始化知识库集合

将语料写入 `self_data/*.txt`，按需调用 [data_storage.py](file:///d:/bishe/data_storage.py) 的
`collection_create(url, collection_name)` 创建 Chroma 集合；集合名固定为
`self_introduction`、`music_analysis`、`music_list` 三个，与路由逻辑对应。
删除集合用 `drop_collection(collection_name)`。

分块与检索用的 Embedding 必须与建库时一致，改 `EMBEDDING_MODEL` 后需重建集合。

### 4. 启动后端

```bash
python main.py
# 或： uvicorn main:app --host localhost --port 8000 --reload
```

后端默认运行在 `http://localhost:8000`。

### 5. 启动前端

```bash
cd music-chatbot-frontend
npm install
npm start
```

前端默认运行在 `http://localhost:3000`，接口地址在
[api.js](file:///d:/bishe/music-chatbot-frontend/src/service/api.js#L4-L9) 中配置（`baseURL: http://localhost:8000`）。

---

## 评测

单条问题走完整 RAG 链路后，输出检索上下文与生成答案的 BLEU、ROUGE-1/2/L 以及 LLM 忠诚度得分：

```bash
python llm_ev.py
# 提示：请输入用户问题: 贝多芬第五交响曲有什么特点？
```

- **BLEU / ROUGE**：把检索上下文当作参考文本，衡量生成答案对其的覆盖程度。
- **忠诚度（0/1/2）**：由 LLM 判断答案是否有参考文本之外的虚构或矛盾，输出 JSON 含 `score` 与 `justification`。

---

## 注意事项

- 会话数据存于内存字典 `sessions`，重启后丢失，且未做持久化与并发保护。
- CORS 当前为 `allow_origins=["*"]`，仅适合本地开发。
- 上传解析仅处理 `.docx`（本质是 zip 内 XML），其它格式不会提取到正文。
- 混合检索中的 BM25 索引在每次检索时从集合内全量文档重建（Chroma 只存稠密向量），
  语料规模较大时会有额外开销；`data_storage.py` 的 `DENSE_WEIGHT` / `SPARSE_WEIGHT` 可调融合权重。
- 向量库已从 Milvus 迁移到 ChromaDB，旧的 `milvus-standalone/` 与 Docker Compose 已移除，
  原 Milvus 中已录入的数据不会自动迁移，需要重新执行 `collection_create` 建库。