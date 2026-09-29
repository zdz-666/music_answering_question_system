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
- **多轮对话**：后端按 `session_id` 维护会话，最多保留最近 20 条消息，生成时注入最近 4 条作为上下文；会话落盘在 SQLite，多 worker 共享、重启不丢。
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
| 会话存储 | SQLite（标准库 `sqlite3`，WAL 模式，多进程共享同一个文件） |
| 检索 | Chroma 稠密检索 + BM25 稀疏检索，经 `EnsembleRetriever` 加权融合，再由 Reranker（SiliconFlow `Qwen/Qwen3-Reranker-8B`）截取前 50% |
| 联网搜索 | Tavily HTTP 接口（用 `httpx` 直调，带超时与 `tenacity` 重试） |
| 分词/评测 | jieba、nltk（BLEU）、rouge-score |
| 前端 | React 19 + Create React App + axios |

---

## 目录结构

```
bishe/
├── main.py                 # FastAPI 入口，路由、文件上传（会话已外置到 session_store.py）
├── rag.py                  # RAG 主链路：查询重写、检索、自反思、答案生成
├── collection_router.py    # LLM 结构化输出做集合路由（含独立测试入口）
├── data_storage.py         # Chroma 集合创建/删除、稠密+BM25 混合检索
├── dynamic_chunk.py        # 基于句子语义相似度的动态分块（SemanticChunker）
├── config.py               # 模型 / 密钥 / Chroma 路径 / 会话库路径的统一配置入口
├── observability.py        # 结构化日志：JSON 格式、request_id、分步耗时、token 成本
├── session_store.py        # 会话持久化：SQLite 建表、消息增删查、20 条裁剪
├── models/                 # 接口层与业务层共用的 Pydantic 数据结构
├── .env.example            # 环境变量模板（复制为 .env 后填写）
├── llm_ev.py               # 评测脚本：BLEU / ROUGE / LLM 忠诚度
├── self_data/              # 示例私人语料（自我介绍、音乐分析、歌单）
├── chroma_db/              # Chroma 持久化数据（运行时生成，不入库）
├── sessions.db             # 会话库（运行时生成，不入库）
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

ChromaDB 以嵌入式方式运行，数据直接落盘到 `CHROMA_PERSIST_DIR`；会话库是标准库 `sqlite3`
写的单个文件 `sessions.db`。**两者都不需要 Docker，也不需要单独启动数据库服务**。

### 1. 后端依赖

```bash
pip install fastapi uvicorn langchain langchain-openai langchain-chroma langchain-classic \
    langchain-community chromadb rank-bm25 httpx tenacity jieba nltk rouge-score \
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
`RERANK_API_KEY`（重排序）、`CHROMA_PERSIST_DIR`（向量库落盘目录，默认 `./chroma_db`）、
`SESSION_DB_PATH`（会话库文件路径，默认 `./sessions.db`）、
`RERANK_TIMEOUT` / `WEB_SEARCH_TIMEOUT` / `HTTP_RETRIES`（外部调用的超时秒数与重试次数）。

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

## 可观测性（结构化日志）

链路有 6 次 LLM 调用与 3 次外部服务调用，为定位瓶颈全部输出结构化日志。

**输出方式**：stdout，每行一条 JSON，非 JSON 日志不混入。级别用 `LOG_LEVEL` 控制。
实现见 [observability.py](file:///d:/bishe/observability.py)。

**公共字段**：`ts`、`level`、`logger`、`request_id`、`event`。
`request_id` 由 [main.py](file:///d:/bishe/main.py#L53-L67) 的中间件按请求生成，存在 `contextvar` 里
贯穿整条调用链，业务函数无需层层传参，并回写到响应头 `X-Request-Id` 便于前端对齐。
RAG 链路被下沉到线程池执行（见「注意事项」），`contextvar` 会随任务一起拷进工作线程，
因此线程内产生的 `llm.usage`、`step.done` 仍带着同一个 `request_id`，也照常计入 `rag.total` 的合计。

**事件类型**

| event | 说明 | 关键字段 |
| --- | --- | --- |
| `step.start` / `step.done` / `step.error` | 单步开始、结束（含耗时）、异常 | `step`、`elapsed_ms`、`input_tokens`、`output_tokens`、`llm_calls`、`cost_usd` |
| `llm.usage` | 每次 LLM 调用结束 | `step`、`model`、`input_tokens`、`output_tokens`、`total_tokens`、`cost_usd` |
| `request.status` | HTTP 响应状态码 | `status_code` |
| `http.retry` | 外部 HTTP 调用失败后即将重试 | `attempt`、`max_attempts`、`error` |
| `rerank.degraded` | 重排服务不可用，退化为检索原序 | `step`、`reason`、`docs_keep` |
| `self_reflection.result` | 自反思过滤后的文本 | `content` |
| `collection.create` / `collection.drop` | 知识库集合操作结果 | `collection`、`result`、`docs` |

**已埋点的步骤**：`request`（请求级）→ `rag.total`（RAG 链路汇总）→
`query_rewriting`、`collection_router`、`vector_search`、`rerank`、`web_search`、
`web_rewriting`、`self_reflection`、`generate`。

嵌套步骤的 token 与成本会向上累加，因此 `rag.total` 一行就是整条链路的合计。

**定位慢点**：按 `elapsed_ms` 排序找出最慢的步骤

```bash
python main.py 2>&1 | python -c "
import sys, json
# 跳过 uvicorn 的纯文本行
rows = [json.loads(l) for l in sys.stdin if l.startswith('{')]
done = [r for r in rows if r.get('event') == 'step.done']
for r in sorted(done, key=lambda x: -x.get('elapsed_ms', 0))[:10]:
    print(f\"{r['elapsed_ms']:>9.1f}ms  {r['step']:<18} rid={r['request_id']}\")
"
```

输出示例（按耗时降序，一眼看出瓶颈在 `generate`）：

```text
   4210.2ms  generate           rid=a1b2c3d4e5f6
    320.5ms  query_rewriting    rid=a1b2c3d4e5f6
     88.0ms  rerank             rid=a1b2c3d4e5f6
```

**成本折算**：`cost_usd` 需要配置单价才非零——在 `.env` 里填 `LLM_INPUT_PRICE_PER_MT` /
`LLM_OUTPUT_PRICE_PER_MT`（每百万 token 单价）。未配置时只记录 token 数。
注意当前只统计**对话模型**，Embedding 的 token 未计入（分块与检索各自会调用 Embedding）。

---

## 注意事项

- 会话数据已外置到 SQLite（[session_store.py](file:///d:/bishe/session_store.py)，库文件 `SESSION_DB_PATH`）：
  所有 worker 读写同一个文件，`uvicorn --workers 2` 不再各存一份，进程重启也不丢。
  建表与切换 WAL 由 [main.py](file:///d:/bishe/main.py#L31-L35) 的 `lifespan` 在启动时执行，幂等。
  并发写靠 WAL（读写可并存）+ `busy_timeout`（写冲突时等待而不是报 `database is locked`）兜住。
  连接按线程复用，避免每次开关连接都触发 WAL checkpoint 并删 `-wal`/`-shm`（Windows 上约 35ms/次），
  复用后单次读写约 0.4ms，因此没有再把会话读写下沉到线程池。
- 会话 id 改为「时间戳 + 随机后缀」（原先用 `len(sessions)` 当后缀，多 worker 下会算出相同 id）。
  传入一个库里不存在的 `session_id` 会新建会话而不是报错——与原实现语义一致。
- 单个会话只保留最近 20 条消息；`DELETE` 清空历史只删消息、保留会话本身，因此清空后再查历史返回
  200 + 空列表，只有从未存在的 `session_id` 才返回 404。
- 耗时的业务函数都是同步实现（LLM、`requests`、Chroma 都没有异步 API），直接在 `async def`
  端点里调用会占住事件循环，同一 worker 上的请求只能排队。现已用 `run_in_threadpool` 下沉到
  线程池：RAG 链路在 [chat](file:///d:/bishe/main.py#L79-L80) / [upload](file:///d:/bishe/main.py#L152-L153)，
  知识库写入在 [三个 knowledge 端点](file:///d:/bishe/main.py#L171-L193)
  （`chunk_document` 与 `add_documents` 内部要调 Embedding，语料大时是秒级）。
  线程上限由 anyio 默认的 40 控制，超出后新请求会排队等待空闲线程。
- 外部调用统一走 `rag._post_json`，都带超时并做指数退避重试：只对超时、网络错误、5xx、429 重试，
  4xx（比如密钥错）直接失败，不浪费尝试。最坏耗时 ≈ 超时秒数 × (`HTTP_RETRIES` + 1)。
- 重排是增强步骤，不是必需项：服务不可用（超时 / 5xx / 返回结构对不上）时退化为混合检索的
  原顺序并截断，本次问答照样出结果，同时打 `rerank.degraded` 警告。联网搜索失败仍会向上抛 500，
  未做降级——搜索结果是主链路输入，静默降级会让答案在无凭据的情况下生成。
- `web_search` 只取 Tavily 返回的第一条结果（`results[0]`），命中为空列表时会抛 `IndexError`；
  这是既有行为，未纳入本次改动。
- CORS 当前为 `allow_origins=["*"]`，仅适合本地开发。
- 上传解析仅处理 `.docx`（本质是 zip 内 XML），其它格式不会提取到正文。
- 混合检索中的 BM25 索引在每次检索时从集合内全量文档重建（Chroma 只存稠密向量），
  语料规模较大时会有额外开销；`data_storage.py` 的 `DENSE_WEIGHT` / `SPARSE_WEIGHT` 可调融合权重。
- 向量库已从 Milvus 迁移到 ChromaDB，旧的 `milvus-standalone/` 与 Docker Compose 已移除，
  原 Milvus 中已录入的数据不会自动迁移，需要重新执行 `collection_create` 建库。
- 应用日志为纯 JSON 行，但 uvicorn 自身的 access log 仍是纯文本且同样写 stdout，两者会混排。
  需要管道里只留 JSON 时，用 `uvicorn main:app --no-access-log` 启动。
- 日志里目前会记录自反思的完整文本（`self_reflection.result`），语料较私密时注意日志落盘范围。