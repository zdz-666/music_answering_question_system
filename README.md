# 乐典音乐助手（Music Chatbot）

一个基于 **RAG（检索增强生成）** 的音乐知识问答系统。后端使用 FastAPI + LangChain 编排
「查询重写 → 智能路由 → 向量/网络检索 → 重排序 → 自反思过滤 → 生成」的完整链路，
前端使用 React 提供聊天界面，向量数据由 ChromaDB 本地持久化。

---

## 功能特性

- **多路检索融合**：同时支持本地知识库检索与联网搜索，走知识库、走网络还是两者都走由 LLM 依据提问自动决定，无需人工开关。
- **检索补充（Retrieval Reflection）**：检索完成后由 LLM 判断信息是否足够，不够就用新语句再检索一轮，最多补充 2 轮。
- **查询重写**：把口语化的用户提问改写为适合向量检索的正式查询；联网搜索前再单独做一次关键词净化。
- **智能路由（Collection Router）**：由 LLM 判断问题应该落到哪个集合，支持多集合同时命中。
- **混合检索 + 重排序**：Chroma 稠密检索与 BM25 稀疏检索按 0.6/0.4 加权融合，再用 Reranker 模型按相关性截取前 50%。
- **自反思过滤（Self-Reflection）**：在生成答案前，先由 LLM 从知识库/网络结果中剔除无关信息。
- **多轮对话**：后端按 `session_id` 维护会话，最多保留最近 20 条消息，生成时注入最近 4 条作为上下文；会话落盘在 SQLite，多 worker 共享、重启不丢。
- **文件问答**：上传 `.pdf` / `.docx`，同时解析正文与内嵌图片——图片由视觉模型（VLM）
  转成中文描述后一起参与问答，扫描页（无文字层）自动整页渲染识别。
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
| 文档解析 | PyMuPDF（PDF 文字 / 内嵌图 / 扫描页）、python-docx + zip（DOCX 正文与图片）、Pillow（图片尺寸） |
| 分词/评测 | jieba、nltk（BLEU）、rouge-score |
| 前端 | React 19 + Create React App + axios |

---

## 目录结构

```
bishe/
├── main.py                 # FastAPI 入口，路由、文件上传（会话已外置到 session_store.py）
├── rag.py                  # RAG 主链路：查询重写、检索、自反思、答案生成
├── collection_router.py    # LLM 结构化输出做集合路由（含独立测试入口）
├── retrieval_planner.py    # LLM 结构化输出做检索决策：通道规划 + 信息是否充分
├── document_loader.py      # 上传文档解析：PDF/DOCX 正文与内嵌图片、VLM 图片描述、描述缓存
├── data_storage.py         # Chroma 集合创建/删除、稠密+BM25 混合检索
├── dynamic_chunk.py        # 基于句子语义相似度的动态分块（SemanticChunker）
├── config.py               # 模型 / 密钥 / Chroma 路径 / 会话库路径的统一配置入口
├── observability.py        # 链路追踪 + 结构化日志：span、JSON 格式、分步输入输出/耗时/token
├── trace_view.py           # 按 request_id 把日志还原成调用树（离线看链路）
├── session_store.py        # 会话持久化：SQLite 建表、消息增删查、20 条裁剪
├── models/                 # 接口层与业务层共用的 Pydantic 数据结构
├── .env.example            # 环境变量模板（复制为 .env 后填写）
├── llm_ev.py               # 评测脚本：BLEU / ROUGE / LLM 忠诚度
├── self_data/              # 示例私人语料（自我介绍、音乐分析、歌单）
├── chroma_db/              # Chroma 持久化数据（运行时生成，不入库）
├── sessions.db             # 会话库（运行时生成，不入库）
├── uploaded_images/        # 上传文档抽出的图片与描述缓存（运行时生成，不入库）
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
   ▼
检索通道规划 (retrieval_planner) ──► 知识库 / 网络搜索 / 两者 / 都不需要
   │
   ├─（含知识库）集合路由 (collection_router) → 查询重写 (query_rewriting)
   │                 └─ Chroma 稠密 + BM25 稀疏混合检索 (k=6) ──► Reranker 取前 50%
   │
   ├─（含网络）  web_rewriting → Tavily 联网搜索（取最相关的 5 条）
   │
   ▼
检索补充循环（最多 MAX_SUPPLEMENT_ROUNDS = 2 轮）
   │  retrieval_sufficiency ──► 信息够不够？不够则给出新语句
   │        └─ 用新语句在首轮通道里再查一遍 → 结果追加去重
   │
   ▼
自反思过滤 (self_reflection) ──► 与问题相关的知识库信息 + 网络信息
   │
   ▼
答案生成 (乐典助手 Prompt + 历史对话 + 上传文件正文及其图片描述)
   │
   ▼
返回答案 & 记录会话
```

通道规划判定「都不需要」时跳过全部检索与自反思，直接由生成步骤凭自身知识作答。

分块策略见 [dynamic_chunk.py](file:///d:/bishe/dynamic_chunk.py)：先按中英文标点切句，逐句向量化，
相邻句余弦相似度低于 `0.7` 处切分，再按 `max_chunk_size` 强制截断并支持重叠。

---

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 发送消息，返回答案与 `session_id` |
| GET | `/api/chat/history/{session_id}` | 查询会话历史（`number` 可选，默认 20） |
| DELETE | `/api/chat/history/{session_id}` | 清空指定会话历史 |
| POST | `/api/upload` | 上传 `.pdf` / `.docx` 文件并提问（含图片描述），其它格式返回 400 |
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
    numpy torch python-multipart python-dotenv pymupdf python-docx pillow
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
`RERANK_TIMEOUT` / `WEB_SEARCH_TIMEOUT` / `HTTP_RETRIES`（外部调用的超时秒数与重试次数）、
`MAX_SUPPLEMENT_ROUNDS`（检索不足时最多追加几轮补充检索，默认 2，设 0 关闭）、
`VLM_MODEL` / `VLM_TIMEOUT`（上传文档的图片描述模型与超时，留空则不生成图片描述）、
`IMAGE_MIN_SIZE`（小于该像素的图片直接丢弃，默认 100）、`UPLOAD_IMAGE_DIR`（抽出的图片与描述缓存目录）、
`LOG_LEVEL` / `TRACE_TEXT_LIMIT`（日志级别与链路追踪里每段输入输出的预览字符上限）。

业务侧统一通过 `config.get_chat_model()`、`config.get_embeddings()`、`config.get_vlm()` 取实例，
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

## 可观测性（链路追踪 + 结构化日志）

链路有 5~12 次 LLM 调用（取决于检索通道与补充检索轮数）与若干次外部服务调用，
为定位瓶颈全部输出结构化日志：
「改写 → 路由 → 检索 → 重排 → 生成」每一步的**输入、输出、耗时、token、成本**都记下来，
并可按 `request_id` 还原成一条调用树。实现见 [observability.py](file:///d:/bishe/observability.py)。

**输出方式**：每行一条 JSON，级别用 `LOG_LEVEL` 控制。

> **日志走 stderr**（`logging.StreamHandler()` 的默认流），不是 stdout。抓日志时注意：
> PowerShell 5.1 下 `python main.py 2>app.log` 会给每行加 `python.exe : ` 前缀，
> 并按控制台宽度硬折行，把 JSON 从物理上切断。稳妥做法是让父进程把 stderr 写进文件句柄
> （`subprocess.run(..., stderr=f)`），或先 `uvicorn main:app --no-access-log` 再重定向，
> 这样文件里才是干净的原始 JSON。

**公共字段**：`ts`、`level`、`logger`、`request_id`、`event`。

### Span 模型

- `request_id` 是一次请求的 trace id，由 [main.py](file:///d:/bishe/main.py#L53-L67) 的中间件按请求生成，
  存在 `contextvar` 里贯穿整条调用链，业务函数无需层层传参，并回写到响应头 `X-Request-Id`
  便于前端对齐。RAG 链路被下沉到线程池执行（见「注意事项」），`contextvar` 会随任务一起拷进
  工作线程，因此线程内产生的 `llm.usage`、`step.done` 仍带着同一个 `request_id`。
- 每个 `log_step(...)` 生成一个 span，带自己的 `span_id`（8 位 hex）与 `parent_span_id`；
  层级从嵌套关系自动推导（`request → rag.total → generate`）。同一份日志里重复出现的同名步骤
  （多个集合各检索一次）也能靠 `span_id` 区分。
- 入参在进入时用 `log_step(step, input=...)` 传入，出参在块内挂到 `span.output`。
  `step.start` 带一份入参，`step.done` 带耗时 / token / 成本 / 入参 / 出参——
  **单行即自描述**：`grep '"step":"generate"' app.log | tail -1` 就能看全这步吃了什么、
  吐出什么、花了多久、烧了多少 token。
- 嵌套步骤的 token 与成本会向上累加，因此 `rag.total` 一行就是整条链路的合计。
- 入参/出参按 `TRACE_TEXT_LIMIT`（默认 500 字符）截断预览：字符串留头尾各半并标注省略了多少字符，
  dict 逐字段截断（保证多字段入参每个字段都看得见），列表只留前 5 项并标注剩余数量；
  同时输出 `input_chars` / `output_chars`，记录**截断前**的原始长度。

**事件类型**

| event | 说明 | 关键字段 |
| --- | --- | --- |
| `step.start` / `step.done` / `step.error` | 单步开始、结束（含耗时）、异常 | `step`、`span_id`、`parent_span_id`、`elapsed_ms`、`input`、`input_chars`、`output`、`output_chars`、`input_tokens`、`output_tokens`、`llm_calls`、`cost_usd` |
| `llm.usage` | 每次 LLM 调用结束 | `step`、`span_id`、`model`、`input_tokens`、`output_tokens`、`total_tokens`、`cost_usd` |
| `request.status` | HTTP 响应状态码 | `status_code` |
| `http.retry` | 外部 HTTP 调用失败后即将重试 | `attempt`、`max_attempts`、`error` |
| `rerank.degraded` | 重排服务不可用，退化为检索原序 | `step`、`reason`、`docs_keep` |
| `web_search.empty` | 联网搜索返回空结果（不再抛 IndexError） | `query` |
| `supplement.assess_failed` / `supplement.retrieve_failed` | 补充检索的审查或再检索失败，已终止补充 | `round`、`query`、`error` |
| `retrieval_planner.failed` | 通道规划模型不可用，退化为「两者都检索」 | `error` |
| `self_reflection.result` | 自反思过滤后的文本 | `content` |
| `collection.create` / `collection.drop` | 知识库集合操作结果 | `collection`、`result`、`docs` |

**已埋点的步骤**：`request`（请求级）→ `rag.total`（RAG 链路汇总）→
`retrieval_planner`、`query_rewriting`、`collection_router`、`vector_search`、`rerank`、
`web_search`、`web_rewriting`、`retrieval_sufficiency`、`self_reflection`、`generate`。

补充检索会让同一 `request_id` 下出现多个同名的 `vector_search` / `web_search` / `query_rewriting` span，
用 `span_id` 区分轮次（与「多个集合各检索一次」是同一套机制）。通道被判定为不需要时，
对应的 `collection_router` / `web_rewriting` 等 span 会**整条不出现**，可据此确认通道确实没被白跑。

### 还原调用树（trace_view.py）

日志是扁平的 JSON 行，[trace_view.py](file:///d:/bishe/trace_view.py) 按 `span_id` / `parent_span_id`
把一条链路重新拼成树，并把每步的耗时、token、成本、输入输出放在一起：

```bash
# 看日志文件里最后一条"已完成"的链路（未指定时默认如此）
python trace_view.py app.log

# 指定 request_id（支持子串）
python trace_view.py app.log --request-id 3f2a1c8e

# 列出日志里所有请求（request_id / 耗时 / 成本 / 入口）
python trace_view.py app.log --list

# 只看耗时与成本，不展开输入输出
python trace_view.py app.log --no-io

# 实时：另开一个窗口发请求，这边边跑边看
python main.py 2>&1 | python trace_view.py
```

输出示例（`git` 风格的树，节点上是耗时 + token + 成本，下面挂输入输出与每次模型调用）：

```text
request_id = 84393f4460fb
request  31207.4ms  3028/2030 tok  $0.000000  7 llm  path=/api/chat  method=POST
   └─ rag.total  31207.4ms  3028/2030 tok  $0.000000  7 llm  session_id=s-1a2b3c4d5e6f
         in [6字符] : 介绍一下肖邦
         out[312字符] : 肖邦是浪漫主义时期的波兰作曲家……
      ├─ retrieval_planner  1421.6ms  143/9 tok  $0.000000  1 llm
      │     in [6字符] : 介绍一下肖邦
      │     out[47字符] : {"use_knowledge_base": true, "use_web_search": true}
      │     llm: deepseek-ai/DeepSeek-V4-Flash  143/9 tok  $0.000000
      ├─ collection_router  1938.1ms  121/9 tok  $0.000000  1 llm
      │     in [6字符] : 介绍一下肖邦
      │     out[29字符] : ["music_analysis", "music_list"]
      │     llm: deepseek-ai/DeepSeek-V4-Flash  121/9 tok  $0.000000
      ├─ query_rewriting  1566.1ms  187/12 tok  $0.000000  1 llm  query_len=6
      │     in [6字符] : 介绍一下肖邦
      │     out[13字符] : 肖邦 生平 作品 音乐风格
      │     llm: deepseek-ai/DeepSeek-V4-Flash  187/12 tok  $0.000000
      ├─ vector_search  0.1ms  collection=music_analysis  k=6
      │     in [13字符] : 肖邦 生平 作品 音乐风格
      │     out[41字符] : ["肖邦的夜曲…", "德彪西的月光…", "肖邦的练习曲…"]
      ├─ vector_search  0.2ms  collection=music_list  k=6
      │     in [13字符] : 肖邦 生平 作品 音乐风格
      │     out[22字符] : ["肖邦作品精选…", "古典钢琴独奏…"]
      ├─ rerank  1174.2ms  model=Qwen/Qwen3-Reranker-8B  docs_in=5  docs_keep=3
      │     in [123字符] : {"query": "肖邦 生平 作品 音乐风格", "documents": [...]}
      │     out[41字符] : ["肖邦的夜曲…", "肖邦的练习曲…", "肖邦作品精选…"]
      ├─ web_rewriting  2426.2ms  96/8 tok  $0.000000  1 llm  query_len=6
      │     in [6字符] : 介绍一下肖邦
      │     out[10字符] : 肖邦 生平 代表作品
      │     llm: deepseek-ai/DeepSeek-V4-Flash  96/8 tok  $0.000000
      ├─ web_search  2048.0ms  provider=tavily  max_results=5
      │     in [10字符] : 肖邦 生平 代表作品
      │     out[612字符] : [{"title": "肖邦 - 维基百科", "url": "https://…", "content": "弗雷德里克·肖邦…"}, ...]
      ├─ retrieval_sufficiency  1611.7ms  512/14 tok  $0.000000  1 llm  round_index=1
      │     in [52字符] : {"question": "介绍一下肖邦", "channel": "knowledge_base+web_search"}
      │     out[39字符] : {"is_sufficient": true, "new_query": ""}
      │     llm: deepseek-ai/DeepSeek-V4-Flash  512/14 tok  $0.000000
      ├─ self_reflection  1815.3ms  342/87 tok  $0.000000  1 llm  has_vector=true  has_web=true
      │     in [286字符] : {"query": "介绍一下肖邦", "vector_result": "肖邦的夜曲…", "web_result": "[1] 肖邦 - 维基百科…"}
      │     out[58字符] : 相关的知识库信息：… 相关的网络信息：…
      │     llm: deepseek-ai/DeepSeek-V4-Flash  342/87 tok  $0.000000
      └─ generate  16398.2ms  1627/1891 tok  $0.000000  1 llm  has_file=false
            in [52字符] : {"question": "介绍一下肖邦", "history_string": "", "file_content": "", "context": "…"}
            out[312字符] : 肖邦是浪漫主义时期的波兰作曲家……
            llm: deepseek-ai/DeepSeek-V4-Flash  1627/1891 tok  $0.000000
```

> 一次请求里 `vector_search` 会出现多次（命中多个集合各检索一次），靠 `collection=` 与
> `span_id` 区分；同一段日志里其它请求的行会被自动忽略。上面这条 `retrieval_sufficiency`
> 判定「信息已足够」（`is_sufficient: true`），因此没有补充检索那一轮。

判定「不够」时，`retrieval_sufficiency` 会给出新语句，后面紧跟着第二轮检索，最后再审查一次
（同一棵树里只截取与补充检索相关的节点，`vector_search` / `rerank` 等首轮节点未展开）：

```text
request_id = a17c9e2b40d5
rag.total  6124.3ms  1322/44 tok  $0.000000  3 llm  session_id=s-77aa11bb22cc
      in [13字符] : 推荐几首适合我口味的曲子
   ├─ retrieval_sufficiency  1590.4ms  486/21 tok  $0.000000  1 llm  round_index=1
   │     in [59字符] : {"question": "推荐几首适合我口味的曲子", "channel": "knowledge_base+web_search"}
   │     out[56字符] : {"is_sufficient": false, "new_query": "我喜欢的 钢琴 曲目 歌单"}
   │     llm: deepseek-ai/DeepSeek-V4-Flash  486/21 tok  $0.000000
   ├─ query_rewriting  1502.8ms  196/11 tok  $0.000000  1 llm  query_len=14
   │     in [14字符] : 我喜欢的 钢琴 曲目 歌单
   │     out[11字符] : 钢琴 曲目 偏好 歌单
   │     llm: deepseek-ai/DeepSeek-V4-Flash  196/11 tok  $0.000000
   ├─ vector_search  0.2ms  collection=music_list  k=6
   │     in [11字符] : 钢琴 曲目 偏好 歌单
   │     out[22字符] : ["肖邦作品精选…", "德彪西意向集…"]
   ├─ rerank  1103.5ms  model=Qwen/Qwen3-Reranker-8B  docs_in=2  docs_keep=1
   │     in [80字符] : {"query": "钢琴 曲目 偏好 歌单", "documents": ["肖邦作品精选…", "德彪西意向集…"]}
   │     out[11字符] : ["肖邦作品精选…"]
   └─ retrieval_sufficiency  1547.1ms  640/12 tok  $0.000000  1 llm  round_index=2
         in [59字符] : {"question": "推荐几首适合我口味的曲子", "channel": "knowledge_base+web_search"}
         out[39字符] : {"is_sufficient": true, "new_query": ""}
         llm: deepseek-ai/DeepSeek-V4-Flash  640/12 tok  $0.000000
```

第二轮用 `round_index=2` 与首轮区分；轮次上限由 `MAX_SUPPLEMENT_ROUNDS` 控制。

**定位慢点**：树里每个节点后面的第一个数字就是该步耗时，顺着根往下看哪一层最重即可；
需要跨请求按耗时排序时，直接读 JSON 行（先按上面说的方式把日志落成干净文件）：

```bash
python -c "
import json
rows = (json.loads(l[l.find('{'):]) for l in open('app.log', encoding='utf-8') if '{' in l)
done = [r for r in rows if r.get('event') == 'step.done']
for r in sorted(done, key=lambda x: -x.get('elapsed_ms', 0))[:10]:
    print('{:>9.1f}ms  {:<18} span={}  rid={}'.format(
        r.get('elapsed_ms', 0), r.get('step', ''), r.get('span_id'), r['request_id']))
"
```

**成本折算**：`cost_usd` 需要配置单价才非零——在 `.env` 里填 `LLM_INPUT_PRICE_PER_MT` /
`LLM_OUTPUT_PRICE_PER_MT`（每百万 token 单价）。未配置时只记录 token 数（示例里全为 0）。
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
  线程池：RAG 链路在 [chat](file:///d:/bishe/main.py#L79-L80)、上传解析与问答在
  [upload](file:///d:/bishe/main.py#L121-L160)，
  知识库写入在 [三个 knowledge 端点](file:///d:/bishe/main.py#L171-L193)
  （`chunk_document` 与 `add_documents` 内部要调 Embedding，语料大时是秒级）。
  线程上限由 anyio 默认的 40 控制，超出后新请求会排队等待空闲线程。
- 外部调用统一走 `rag._post_json`，都带超时并做指数退避重试：只对超时、网络错误、5xx、429 重试，
  4xx（比如密钥错）直接失败，不浪费尝试。最坏耗时 ≈ 超时秒数 × (`HTTP_RETRIES` + 1)。
- 重排是增强步骤，不是必需项：服务不可用（超时 / 5xx / 返回结构对不上）时退化为混合检索的
  原顺序并截断，本次问答照样出结果，同时打 `rerank.degraded` 警告。联网搜索失败仍会向上抛 500，
  未做降级——搜索结果是主链路输入，静默降级会让答案在无凭据的情况下生成。
- `web_search` 取 Tavily 返回的前 5 条（`TAVILY_MAX_RESULTS`），交给 prompt 前由
  `format_web_items` 渲染成「[序号] 标题 / URL / 内容」的文本块；命中空结果时只打一条
  `web_search.empty` 警告并退化为空列表，不再抛 `IndexError`。
- 补充检索（`retrieval_sufficiency` → 再检索，≤ `MAX_SUPPLEMENT_ROUNDS` 轮）是**可选增强**：
  审查模型异常、补充检索异常、或给出的新语句为空/与已用语句重复时，都会立即终止补充并
  保留已有结果继续生成，不会把整条问答打成 500。轮数用 `for` 循环硬封顶，不存在无限循环。
  通道规划模型不可用时退化为「知识库与网络都检索」。每个 `request_id` 下最多约 12 次 LLM 调用，
  延迟上限由 `MAX_SUPPLEMENT_ROUNDS` 控制。
- CORS 当前为 `allow_origins=["*"]`，仅适合本地开发。
- 上传解析支持 `.pdf`（PyMuPDF）与 `.docx`（python-docx + zip）；旧版 `.doc` 与其它格式返回 400。
  解析走 [document_loader.py](file:///d:/bishe/document_loader.py)：正文按页/段落取文本，内嵌图片按
  「尺寸过滤（< `IMAGE_MIN_SIZE` 丢弃）→ MD5 去重 → VLM 描述 → 描述缓存落盘」处理；页面无文字层
  但有图时判定为扫描页，整页按 150dpi 渲染成一张图交给 VLM。未配置 `VLM_MODEL`、或单张图描述失败时，
  该位置降级为占位文本（`[图片：未配置 VLM_MODEL…]` / `[图片：描述生成失败]`），不会让整篇上传失败。
  图片描述以独立段落（`【图片内容（第 N 页）】…`）拼进 `file_content`，因此后续接知识库时应**单独成 chunk**，
  避免与正文混切稀释检索精度。
- 混合检索中的 BM25 索引在每次检索时从集合内全量文档重建（Chroma 只存稠密向量），
  语料规模较大时会有额外开销；`data_storage.py` 的 `DENSE_WEIGHT` / `SPARSE_WEIGHT` 可调融合权重。
- 向量库已从 Milvus 迁移到 ChromaDB，旧的 `milvus-standalone/` 与 Docker Compose 已移除，
  原 Milvus 中已录入的数据不会自动迁移，需要重新执行 `collection_create` 建库。
- 应用日志为纯 JSON 行且写 **stderr**，uvicorn 自身的 access log 是纯文本且写 stdout，
  两者默认不在同一流里；用 `2>&1` 合流时会混排，需要管道里只留 JSON 时用
  `uvicorn main:app --no-access-log` 启动。`trace_view.py` 会跳过非 JSON 行。
- 链路追踪会把每步的输入输出按 `TRACE_TEXT_LIMIT`（默认 500 字符）**截断**后写进日志，
  未截断的原文不落盘；但自反思（`self_reflection.result`）与检索命中的文档片段仍会以预览形式
  进入日志，语料较私密时注意日志的落盘与采集范围。
- `self_reflection` 这一步同时保留了旧的独立事件 `self_reflection.result` 与新的 span 出参，
  两者内容重复；旧事件是为了兼容既有日志检索习惯，未删除。