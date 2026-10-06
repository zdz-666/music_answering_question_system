# 乐典音乐助手（Music Chatbot）

一个基于 **RAG（检索增强生成）** 的音乐知识问答系统。后端使用 FastAPI + LangChain 编排
「查询重写 → 智能路由 → 向量/网络检索 → 重排序 → 自反思过滤 → 生成」的完整链路，
前端使用 React 提供聊天界面。知识库向量数据由 ChromaDB 本地持久化，
**会话与三层记忆以 Redis 为准**（Redis 8 自带的 RediSearch 向量索引承载情景记忆与用户画像）。

---

## 功能特性

- **多路检索融合**：同时支持本地知识库检索与联网搜索，走知识库、走网络还是两者都走由 LLM 依据提问自动决定，无需人工开关。
- **检索补充（Retrieval Reflection）**：检索完成后由 LLM 判断信息是否足够，不够就用新语句再检索一轮，最多补充 2 轮。
- **查询重写**：把口语化的用户提问改写为适合向量检索的正式查询；联网搜索前再单独做一次关键词净化。
- **智能路由（Collection Router）**：由 LLM 判断问题应该落到哪个集合，支持多集合同时命中。
- **混合检索 + 重排序**：Chroma 稠密检索与 BM25 稀疏检索按 0.6/0.4 加权融合，再用 Reranker 模型按相关性截取前 50%。
- **自反思过滤（Self-Reflection）**：在生成答案前，先由 LLM 从知识库/网络结果中剔除无关信息。
- **多轮对话 + 三层记忆**：会话以 Redis 为准（多 worker 共享、重启不丢），并在此基础上分三层记忆——**工作记忆**（最近 4 条消息 + 超过阈值后由 LLM 压缩出的滚动摘要，Redis list + hash）、**情景记忆**（同一 `user_id` 的跨会话历史问答，Redis RediSearch 按语义相似度检索）、**用户画像**（从对话中抽取的长期偏好与实体，同样存在 RediSearch 索引里）。三层同源，过期与备份只有一套策略。
- **文件问答**：上传 `.pdf` / `.docx`，或直接上传一张图片（`.png` / `.jpg` / `.jpeg` / `.gif` / `.webp`）
  单独提问。文档会同时解析正文与内嵌图片——图片由视觉模型（VLM）
  转成中文描述后一起参与问答，扫描页（无文字层）自动整页渲染识别。
  解析与生成走**异步任务队列**（有 Redis 时）：接口立刻返回 `task_id`，前端轮询进度，
  HTTP 连接不必为几十张图的 PDF 干等几分钟。
- **知识库动态写入**：可在前端直接录入「个人信息 / 音乐理解 / 歌单」三类私人数据。
- **Redis 一栈到底**：会话、三层记忆、缓存、限流、上传队列共用一个 Redis 8 实例。**会话与记忆是正确性依赖**，`REDIS_URL` 必须配置，连不上时启动即报错（不会静默降级成「这是一段新对话」）；**缓存 / 限流 / 队列是可选的**，Redis 不可用时自动失效，链路照常跑。
- **四层缓存（可选）**：按「输入是否唯一决定输出」分四层缓存——嵌入向量、查询重写与集合路由结果、
  单集合的检索结果、最终答案；知识库写入时按集合版本号定向失效。未配 `REDIS_URL` 时自动降级为不缓存，链路照常跑。
  实测重复提问时整条链路从 49.3s 降到 0.002s。
- **接口限流（可选）**：`/api/chat` 与 `/api/upload` 走 Redis 令牌桶限流（Lua 保证原子），
  超限返回 `429` 并带 `Retry-After`；按 `user_id` 计数，没有 `user_id` 时退回客户端 IP
  （否则清空 `localStorage` 即可绕过）。Redis 不可用时**一律放行**，不会因为限流组件故障拒掉正常请求。
- **离线评测**：内置 BLEU、ROUGE 与 LLM 忠诚度打分脚本（`llm_ev.py`），以及在 RGB 中文基准
  `zh_refine.json`（300 条）上的端到端评测（`rgb_eval.py`，当前准确率 91%）。

---

## 技术栈

| 层次 | 技术 |
| --- | --- |
| 后端框架 | FastAPI + Uvicorn |
| LLM 编排 | LangChain（`langchain-openai`、`langchain-chroma`、`langchain-classic`、`langchain-community`） |
| 向量数据库 | ChromaDB 1.4.0（知识库三集合，本地持久化，无需独立服务或 Docker） |
| 会话存储 | Redis（key 前缀 `sess:`：meta hash + msgs list + 滚动摘要；消息追加与压缩提交走 Lua 保证原子） |
| 记忆存储 | Redis RediSearch 向量索引（HNSW / FLOAT32 / COSINE，一份 hash 同时存正文、时间戳与向量） |
| 缓存 | Redis（可选降级：未配 `REDIS_URL` 或连不上时不缓存，链路照常跑） |
| 限流 | Redis 令牌桶（Lua 脚本原子扣减；Redis 不可用时放行） |
| 任务队列 | Redis 列表（`RPUSH` 入队 + `BLPOP` 出队，由独立进程 `worker.py` 消费；Redis 不可用时上传退回同步） |
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
├── rag.py                  # RAG 主链路：查询重写、检索、自反思、答案生成、上传文件问答流程
├── worker.py               # 上传任务消费者：独立进程从队列取任务，跑「解析 → 检索生成 → 落库」
├── tasks.py                # 任务队列原语：入队/取任务/状态读写、暂存文件、结果与 TTL
├── ratelimit.py            # 接口限流：Lua 令牌桶、按 user_id/IP 计数、Redis 不可用时放行
├── collection_router.py    # LLM 结构化输出做集合路由（含独立测试入口）
├── retrieval_planner.py    # LLM 结构化输出做检索决策：通道规划 + 信息是否充分
├── document_loader.py      # 上传解析：PDF/DOCX 正文与内嵌图、独立图片、VLM 图片描述、描述缓存
├── data_storage.py         # Chroma 集合创建/删除、稠密+BM25 混合检索
├── dynamic_chunk.py        # 基于句子语义相似度的动态分块（SemanticChunker）
├── fixed_chunk.py          # 固定长度 + 重叠的等步长切分（FixedChunker），用于对比切分方式
├── config.py               # 模型 / 密钥 / Chroma 路径 / Redis（会话与记忆）的统一配置入口
├── cache.py                # Redis 连接与缓存层：键前缀、嵌入/检索/答案缓存、集合版本失效、连接降级
├── ratelimit.py            # 接口限流：Lua 令牌桶、按 user_id/IP 计数、Redis 不可用时放行
├── observability.py        # 链路追踪 + 结构化日志：span、JSON 格式、分步输入输出/耗时/token
├── trace_view.py           # 按 request_id 把日志还原成调用树（离线看链路）
├── session_store.py        # 会话持久化：Redis 键设计、消息增删查、滚动摘要与压缩认领（Lua 保证原子）
├── memory/                 # 三层记忆
│   ├── __init__.py         #   对外入口：build_context（取记忆）/ after_turn（写记忆）
│   ├── working.py          #   工作记忆：消息超阈值时用 LLM 压缩成滚动摘要
│   ├── ft_index.py         #   RediSearch 向量索引封装：建索引、写入、KNN 检索、按用户计数
│   ├── episodic.py         #   情景记忆：跨会话问答的读写与「相似度 + 时间近因性」排序
│   └── profile.py          #   用户画像：长期偏好/实体的抽取与检索
├── models/                 # 接口层与业务层共用的 Pydantic 数据结构
├── .env.example            # 环境变量模板（复制为 .env 后填写）
├── llm_ev.py               # 评测脚本：BLEU / ROUGE / LLM 忠诚度
├── rgb_eval.py             # RGB 中文基准评测：合并 positive/negative → 切块 → 走完整 RAG 链路
├── eval_results/           # 评测逐条明细与汇总（运行时生成，不入库）
├── self_data/              # 示例私人语料（自我介绍、音乐分析、歌单）
├── chroma_db/              # Chroma 持久化数据（知识库三集合，运行时生成，不入库）
├── uploaded_images/        # 上传文档抽出的图片与描述缓存（运行时生成，不入库）
├── uploaded_files/         # 上传任务排队期间的暂存文件（worker 处理完即删，不入库）
├── .redis/                 # Redis 容器数据卷（会话 / 记忆 / 缓存都在这，运行时生成，不入库；容器启动命令见下文）
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
答案生成 (乐典助手 Prompt + 历史对话 + 三层记忆 + 上传文件正文及其图片描述)
   │
   ▼
返回答案 & 记录会话
```

通道规划判定「都不需要」时跳过全部检索与自反思，直接由生成步骤凭自身知识作答。

### 文件上传（异步任务）

```
POST /api/upload ──► 限流 ──► 写暂存文件 + 入队（Redis RPUSH）──► 202 {status:"queued", task_id}
                                        │
          worker.py ── BLPOP 取任务 ──► 解析（PyMuPDF / DOCX / VLM 逐张识图）
                                        └─► 三层记忆 → 检索生成 → 落会话与记忆 → 结果写回任务记录
                                        │
          前端轮询 GET /api/task/{task_id} ──► {status: queued|running|done|failed, answer, …}
```

没有 Redis 时 `tasks.submit()` 返回 `None`，`/api/upload` 就地把同一套流程跑完，
返回 `200 {status:"done", answer}` —— 队列是可选设施，不能因为没装 Redis 就让上传不可用。

分块策略见 [dynamic_chunk.py]：先按中英文标点切句，逐句向量化，
相邻句余弦相似度低于 `0.7` 处切分，再按 `max_chunk_size` 强制截断并支持重叠。
另有一种不做语义判断、按固定长度等步长（可带重叠）切分的 [fixed_chunk.py](file:///d:/bishe/fixed_chunk.py)，
两者接口一致，评测时可切换以对比切分方式的影响。

### 会话存储（Redis）

实现在 [session_store.py]。会话原先落在本地 SQLite 文件，
换成 Redis 之后除了继续满足「多 worker 共享、进程重启不丢」，还多了一层：API 进程与 worker 进程
不必再共享同一个文件系统，会话与记忆同源，过期、容量、备份只有一套策略。

| 键 | 类型 | 内容 |
| --- | --- | --- |
| `{prefix}:sess:{id}:meta` | HASH | `created_at` / `updated_at` / `next_id`（消息自增 id） |
| `{prefix}:sess:{id}:msgs` | LIST | 消息，元素是 JSON `{id, role, content, timestamp}` |
| `{prefix}:sess:{id}:sum` | STRING | 滚动摘要 |
| `{prefix}:sess:{id}:claim` | STRING | 压缩认领（`SET NX PX`，值 = 认领时刻） |

- **会话是正确性依赖，不是缓存**：缓存 / 限流 / 队列拿不到 Redis 可以降级，会话不行——
  降级只会把「读不到历史」伪装成「这是一段新对话」，属于静默的数据错误。因此这里取连接用
  `cache.require_client()`，拿不到就抛错（缺 `REDIS_URL` 或连不上时，`init_db()` 在启动阶段即报错）。
- **两段 Lua 保证原子**：追加消息要「取号 → 入队 → 裁剪 → 续期」四步，拆开会有两个 worker 拿到同一号、
  或消息顺序与 id 顺序不一致的竞态；提交压缩要「写摘要 → 删消息 → 释放认领」原子完成，
  否则会出现「摘要更新了但消息没删」这类中间态。
- **TTL**：默认 `SESSION_TTL_SECONDS`（30 天），每次读写都续期，活跃会话不会过期。
- **硬上限兜底**：消息条数超过 `MAX_MESSAGES_HARD`（200）时直接裁掉最旧的，正常路径永远不该触发
  （压缩会先把消息降回 4 条），一旦触发会打 `messages.hard_trim` 警告暴露问题。

### 三层记忆

记忆分三层，各自独立、互不阻塞（任一层失败只打 warning，不影响回答返回）。
记忆只在生成 prompt 里作为独立段落注入，检索链路与评测路径（`get_result_evaluate`）完全不感知。

| 层 | 存储 | 内容 | 读写时机 |
| --- | --- | --- | --- |
| 工作记忆 | Redis `sess:{id}:meta` / `:msgs` / `:sum` | 最近 4 条消息 + 一段滚动摘要 | 每轮直接读；消息超过 `HISTORY_WINDOW + COMPRESS_BATCH`（14）条时由 LLM 把最旧一批并入摘要并删除 |
| 情景记忆 | Redis RediSearch 索引 `{prefix}:memidx:episodic_memory` | 同一 `user_id` 的跨会话历史问答 | 每轮写一条；回答前按「语义相似度 + 时间近因性」加权排序，排除当前会话 |
| 用户画像 | Redis RediSearch 索引 `{prefix}:memidx:user_profile` | 长期偏好与实体（一条一个原子事实） | 每轮由 LLM 抽取（失败降级为空）；回答前按语义相似度检索 |

**情景记忆的时间衰减**：同样相关时，刚聊过的内容优先于很久以前的。每条记忆的 `ts`
（ISO 时间）参与打分，最终得分把两项加权求和：

```
recency_score = exp(-decay_factor * age_hours / 24)     # decay_factor 默认 0.1
final_score   = (1 - w) * 相似度 + w * recency_score    # w 默认 0.3
```

`decay_factor = 0.1` 时，1 天前约 0.90、1 周前约 0.50、1 个月前约 0.05。

相似度直接由 RediSearch 的 KNN 得到：索引按 `DISTANCE_METRIC COSINE` 建，`FT.SEARCH ... =>[KNN k @vec $q AS score]`
返回的 `score` 是**余弦距离**，`相似度 = 1 - score`（负相关截到 0）。这与原先 Chroma 的
`hnsw.space = l2` + 平方距离折算（`cos = 1 - d/2`）不同——后者要专门绕开 langchain 的
`1 - d/√2`（它假设 d 是普通 L2 距离，套在平方距离上会恒为负）。换成 RediSearch 后这段折算消失，
代价是索引维度由**第一次写入的向量**决定（`FT.CREATE` 的 `DIM` 取 `len(vector)`），换嵌入模型需重建索引。

折算出的绝对余弦还会在**候选集内做一次 min-max 归一化**再参与加权：本模型对长短文本的余弦值被
压在很窄的带里（实测同主题约 0.25、不同主题约 0.20，只差 0.055），绝对量纲下相似度最多只能贡献
`0.7×0.055 ≈ 0.039` 分，会被 `0.3×1.0 = 0.3` 的近因性完全淹没，变成「几乎只看时间」。归一化后两项
同在 [0,1]，`w` 才真正控制「相似度 vs 时间」的平衡（候选只有一个或全部同分时取 1.0，排序交给近因性）。

检索时会多取候选再重排（`max(4k, k+5)` 条），否则只按距离截前 k 条的话，被近因性提上来的旧记忆
根本没机会进入候选集。每次重排会打一条 `memory.episodic_rank` 日志，带 `cos_band` 与各候选的
`final / relevance / cosine / recency`，便于观察权重是否合适。时间戳缺失或损坏的记录按「最旧」处理
（近因性 0 分），不会靠新鲜度占便宜。

- **身份隔离**：前端首次访问生成 `user_id` 并存入 `localStorage`（`music_rag_user_id`），随请求带给后端。
  「跨会话」指同一 `user_id` 下的多个 `session_id`。请求不带 `user_id` 时整个记忆层跳过，
  因此 `llm_ev.py` / `rgb_eval.py` 等离线评测不受影响。
- **为什么向量用 RediSearch 而不是别的**：一份 hash 同时存正文、时间戳与向量，KNN 命中时一次往返就把正文取回来，
  不必额外维护「元素 id → 属性」的旁路结构，也就没有两者的同步问题；向量按 float32 原样存，没有量化误差；
  按 `uid` 做 TAG 过滤是服务端行为，别人的记忆根本不会进入候选集。
- **TAG 值先 md5 再进索引**：`user_id` / `session_id` 都来自客户端，可以是任意字符串，而 RediSearch 的 TAG
  查询要转义 `{ } $ \ |` 等字符并按分隔符切分。先摘要就不需要任何转义，索引与查询两侧永远一致；
  原始值另存普通字段（`user` / `session`）用于排查与读回。读回的必须是**原始** session_id——
  否则调用方拿它当 `exclude_session_id` 回传时会被再摘要一次，过滤条件永远匹配不上。
- **压缩走两阶段**：`claim_compression`（`SET NX PX` 抢压缩权）→ LLM 生成摘要 → `commit_compression`
  （一段 Lua 原子完成「写摘要 + 按 `id <= max_id` 弹消息 + 释放认领 + 续期」）。
  LLM 调用不在脚本内，Lua 保证多 worker 下同一会话不会被重复压缩，认领 60s 过期以兜住进程崩溃。
  按 `id` 而非「前 N 条」删除，压缩期间新插入的消息（id 更大）不会被误删。
- **确定性文档 id**：情景记忆用 `md5(user_id | session_id | 问答)`，画像用 `md5(user_id | 事实文本)`，
  落到同一个 `memdoc:*` hash 上重复写即覆盖，天然幂等去重。
- 可调参数：`EPISODIC_TOP_K`、`PROFILE_TOP_K`、`MEMORY_SNIPPET_CHARS`、`EPISODIC_DECAY_FACTOR`、`EPISODIC_RECENCY_WEIGHT`、`EPISODIC_COLLECTION`、`PROFILE_COLLECTION`（见 `.env.example`）。记忆本身不设 TTL，与原 Chroma 实现一致。

### 缓存（Redis）

缓存按「输入是否唯一决定输出」从低风险到高风险分四层，统一收口在 [cache.py](file:///d:/bishe/cache.py)
（键前缀、序列化、降级、失效都在这一个文件里）。命中情况直接进链路日志：`query_rewriting` /
`collection_router` / `vector_search` 三个 span 都多了一个 `cached` 字段。

| 层 | 缓存对象 | 键的组成 | TTL | 失效方式 |
| --- | --- | --- | --- | --- |
| ① | `embed_query` 算出的向量 | 文本摘要 | 1 天 | 只由文本决定，无需失效 |
| ② | 查询重写 / 集合路由结果 | 输入文本摘要 | 1 小时 | TTL |
| ③ | 单集合的「混合检索 + 重排」结果 | 集合名 + 集合版本 + 改写后的查询 + k | 1 小时 | 知识库写入时集合版本 +1，旧键立刻失联 |
| ④ | 最终答案 | 问题摘要 | 5 分钟 | TTL |

- 嵌入缓存包在 `config.get_embeddings()` 里（[config.py]），
  一次覆盖 Chroma 检索、分块、记忆写入全部调用方。`embed_documents` 不走缓存 —— 建库时每段文本都不相同，
  缓存只白占内存不省时间。
- ④ 只在**没有历史、没有长期记忆、没有上传文件**时才启用（[rag.py]）：
  其余情况下同一个问题在不同会话、不同画像下答案本来就不同，拿问题当键会把 A 会话的答案喂给 B 会话。
- 键统一带 `REDIS_KEY_PREFIX`（默认 `musicrag:v1`），改版本号即可让旧格式缓存整体作废。
- 失效用**版本号**而不是 `SCAN` 删键：集合级版本号自增后旧键再也拼不出来，剩下的靠 TTL 自然过期，
  既不阻塞遍历，也不会漏删。
- 值只存 JSON 与 float32 二进制，不用 pickle —— 缓存是可丢弃的派生物，不该在依赖升级后因为反序列化
  失败把进程带崩。4096 维向量存 float32 是 16 KB，存 JSON 数组要 80 KB。
- 同步 redis 客户端会阻塞事件循环，因此缓存调用都发生在已经下沉线程池的**同步**链路里
  （`rag.get_result`、`memory.*` 都由 `run_in_threadpool` 执行）。
- Redis 连不上时只在第一次打一条 `cache.unavailable` 警告，之后静默降级；连接失败不会被记住，
  Redis 起来后不重启进程就能恢复。

### 接口限流

实现在 [ratelimit.py]，复用缓存那一份 Redis 连接配置。

| 接口 | 容量（可突发次数） | 补充速率 | 含义 |
| --- | --- | --- | --- |
| `/api/chat` | 20 | 0.2 次/秒 | 平时约 12 次/分钟，最多连打 20 次 |
| `/api/upload` | 5 | 0.05 次/秒 | 平时约 3 次/分钟，最多连打 5 次 |

- **为什么用 Lua**：一次限流要做「读令牌 → 按时间差补令牌 → 扣令牌 → 写回」四步，
  拆成多次 Redis 调用就会出竞态——两个并发请求可能读到同一份「够用」的令牌双双放行。
  Redis 单线程执行 Lua 脚本，正好把这一串动作压成一次原子调用。
- **令牌桶而不是固定窗口**：桶按距上次访问的时间差惰性补充 `elapsed × refill_per_sec`
  （补到容量为止），不会像固定窗口那样在窗口切换的瞬间放行两倍流量；桶第一次出现时直接装满，
  新用户/新 IP 不会一上来就被卡。
- **自动回收**：桶是一个 hash（剩余令牌 `tokens` + 上次补充时刻 `ts`），`PEXPIRE` 设成
  「桶重新装满的时间 + 60s」，长时间没人访问就自己消失，不必清理。
- **超限响应**：`429`，`detail` 写明还要等几秒，响应头带 `Retry-After`。
- **限流主体**：优先用前端存在 `localStorage` 的 `user_id`（桶键 `u:<id>`），
  没有则退回客户端 IP（桶键 `ip:<host>`）——不退到 IP 的话，清空 `localStorage` 就能绕过限流。
  两者是不同的桶，互不牵连。
- **fail-open**：Redis 连不上或脚本调用报错时**一律放行**，只打一条 `ratelimit.error` 警告。
  限流是保护自己的措施，不是正确性依赖；保护层自己挂了却把正常请求也拒掉，是把可用性
  换成了理论上的安全，不划算。
- `/api/upload` 的限流放在 `file.read()` **之前**，超限的请求在读进内存、调 VLM 之前就被拦下。

### 异步上传任务

实现在 [tasks.py]（队列原语）与 [worker.py]（消费者）。

- **为什么要排队**：一次上传要先用 PyMuPDF/DOCX 解析正文，再逐张把内嵌图交给 VLM 出描述，
  然后才走完整的 RAG 链路。几十张图的 PDF 能跑几分钟，同步处理意味着这个 HTTP 连接
  一直挂着，Nginx/浏览器都可能先超时断开，而活还在照样跑。
- **为什么是独立进程**：FastAPI 的 `BackgroundTasks` 仍在同一个进程里跑，照样占着线程池，
  进程重启任务就没了。独立 worker 可以和 API 分头重启，队列里的任务不丢，
  也能按需多开几个进程一起消费。
- **队列就是 Redis 列表**：`RPUSH` 入队、`BLPOP` 出队，天然 FIFO，不引额外中间件；
  任务状态另存一个 hash（`status` / `payload` / `result` / `error` / `request_id`），带 TTL 自动回收。
- **文件不进 Redis**：payload 里只放暂存文件路径（`UPLOAD_SPOOL_DIR`）。几十 MB 的 PDF 塞进
  同一个 Redis 实例，会挤掉缓存、让 AOF 重写变大；worker 处理完（成功或失败）都会删掉暂存文件。
- **链路可追踪**：入队时把当时 HTTP 请求的 `request_id` 一起存进任务记录，worker 取到任务后
  把它接回 `contextvar`，因此这次上传的日志在 `python trace_view.py app.log --request-id <id>`
  里仍是完整的一棵树（跨进程）。
- **session 与记忆写在 worker 侧**：`add_message` / `memory.after_turn` 与生成在同一处，
  顺序和原先的同步实现完全一致；worker 启动时自己 `init_db()`，不依赖 API 进程先建表。
- **任务状态**：`queued → running → done / failed`。失败时 `error` 里带原因（如格式不支持），
  前端直接展示；任务记录默认存活 `TASK_TTL_SECONDS`（2 小时），过期后轮询返回 404。
- **不做重投递**：worker 崩了任务会停在 `running` 并随 TTL 消失。个人项目上任务重投递要引入
  确认-超时-重排的完整机制，复杂度远大于收益。

---

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 发送消息，返回答案与 `session_id` |
| GET | `/api/chat/history/{session_id}` | 查询会话历史（`number` 可选，默认 20） |
| DELETE | `/api/chat/history/{session_id}` | 清空指定会话历史 |
| POST | `/api/upload` | 上传 `.pdf` / `.docx` / 图片（`.png` `.jpg` `.jpeg` `.gif` `.webp`）并提问（含图片描述），其它格式返回 400。有 Redis 时返回 `202` + `task_id`（202 表示已入队，不是已完成），无 Redis 时退回同步处理、返回 `200` + 答案 |
| GET | `/api/task/{task_id}` | 轮询上传任务：`status` 为 `queued` / `running` / `done` / `failed`，`done` 带答案，`failed` 带 `error`；任务不存在或已过期返回 404 |
| POST | `/api/knowledge/self-introduction` | 写入个人信息 |
| POST | `/api/knowledge/music-analysis` | 写入音乐理解 |
| POST | `/api/knowledge/music-list` | 写入歌单 |

服务启动后可在 `http://localhost:8000/docs` 查看 Swagger 文档。

两个高成本接口 `/api/chat` 与 `/api/upload` 受令牌桶限流保护（见「接口限流」），
超限时返回 `429` 与 `Retry-After`；其余接口不限流。

---

## 快速开始

**Redis 是必须的**：会话与三层记忆都以它为准，没配好 `REDIS_URL` 启动就会报错（见第 4 步）。
知识库的 ChromaDB 仍以嵌入式方式运行，数据直接落盘到 `CHROMA_PERSIST_DIR`，不需要单独启动服务。
缓存、限流与上传任务队列复用同一个 Redis 实例，但它们仍是可选的：
Redis 不可用时缓存退化成每次都算、限流一律放行、上传退回同步处理，会话与记忆之外的链路照常工作。

### 1. 后端依赖

```bash
pip install fastapi uvicorn langchain langchain-openai langchain-chroma langchain-classic \
    langchain-community chromadb rank-bm25 httpx tenacity jieba nltk rouge-score \
    numpy torch python-multipart python-dotenv pymupdf python-docx pillow redis
```

### 2. 配置模型与密钥

模型名、`api_key`、`base_url`、Chroma 存储路径、Redis 地址全部集中在 [config.py](file:///d:/bishe/config.py)，
通过环境变量读取，不再散落在各业务文件里：

```bash
cp .env.example .env    # 然后填写 .env
```

`.env` 已被 [.gitignore]排除，密钥不会进入 git。
[config.py]只保存变量名与安全的默认值，不含任何真实密钥。

需要填写的变量见 [.env.example]：`OPENAI_API_KEY` / `OPENAI_BASE_URL` /
`LLM_MODEL` / `EMBEDDING_MODEL`（对话与向量模型，OpenAI 兼容接口）、`TAVILY_API_KEY`（联网搜索）、
`RERANK_API_KEY`（重排序）、`CHROMA_PERSIST_DIR`（知识库向量库落盘目录，默认 `./chroma_db`）、
`RERANK_TIMEOUT` / `WEB_SEARCH_TIMEOUT` / `HTTP_RETRIES`（外部调用的超时秒数与重试次数）、
`MAX_SUPPLEMENT_ROUNDS`（检索不足时最多追加几轮补充检索，默认 2，设 0 关闭）、
`VLM_MODEL` / `VLM_TIMEOUT`（上传文档的图片描述模型与超时，留空则不生成图片描述）、
`IMAGE_MIN_SIZE`（**文档内嵌图**小于该像素直接丢弃，默认 100；用户单独上传的图片不受此限制）、`UPLOAD_IMAGE_DIR`（抽出的图片与描述缓存目录）、
`REDIS_URL`（**必填**，会话与记忆的正确性依赖；格式 `redis://:密码@localhost:6379/0`）、
`REDIS_KEY_PREFIX`（缓存 / 会话 / 记忆共用的版本前缀）、`SESSION_TTL_SECONDS`（会话键的存活秒数，默认 30 天）、
`CACHE_TTL_*`（各层缓存的存活秒数）、
`RATE_LIMIT_ENABLED`（限流总开关，置 false 关闭）、`RATE_LIMIT_CHAT_*` / `RATE_LIMIT_UPLOAD_*`
（两类接口令牌桶的容量与每秒补充速率）、
`TASK_TTL_SECONDS`（上传任务记录存活秒数，默认 7200）、`UPLOAD_SPOOL_DIR`（上传任务排队期间的暂存目录）、
`LOG_LEVEL` / `TRACE_TEXT_LIMIT`（日志级别与链路追踪里每段输入输出的预览字符上限）。

业务侧统一通过 `config.get_chat_model()`、`config.get_embeddings()`、`config.get_vlm()` 取实例，
因此换模型只需改一处。缺失变量会抛出带操作提示的 `RuntimeError`，
而不是在调用 API 时才报出难懂的 401。

### 3. 初始化知识库集合

将语料写入 `self_data/*.txt`，按需调用 [data_storage.py]的
`collection_create(url, collection_name)` 创建 Chroma 集合；集合名固定为
`self_introduction`、`music_analysis`、`music_list` 三个，与路由逻辑对应。
删除集合用 `drop_collection(collection_name)`。

分块与检索用的 Embedding 必须与建库时一致，改 `EMBEDDING_MODEL` 后需重建集合。

### 4. 启动 Redis（必须）

会话、三层记忆、缓存、限流、上传任务队列都依赖这一个 Redis 实例。

**镜像必须带 RediSearch 模块**：情景记忆与用户画像用的是向量索引（`FT.CREATE` / `FT.SEARCH`），
官方 `redis:8-alpine` 自带（`redisearch` / `vectorset` / `timeseries` / `bf` / `ReJSON` 都在）。
老版本或没带 `search` 模块的精简镜像会让记忆写入失败，因此**不要用 Redis 7 及以下**。

```powershell
mkdir D:\bishe\.redis
docker run -d --name music-rag-redis -p 6379:6379 `
  -v D:\bishe\.redis:/data `
  --restart unless-stopped `
  redis:8-alpine redis-server --appendonly yes --requirepass 你的密码
```

- `-v D:\bishe\.redis:/data` 把数据卷放到 D 盘（该目录已在 `.gitignore` 里），容器重建也不丢；
  `--appendonly yes` 开启 AOF，Redis 重启后会话、记忆与缓存都还在；`--restart unless-stopped`
  让它随 Docker Desktop 自动拉起。
- 这只解决了**数据**的位置。Docker Desktop 自己的虚拟磁盘（镜像与容器可写层）默认仍在
  `C:\Users\<用户名>\AppData\Local\Docker\wsl`，要挪到 D 盘得去 Settings → Resources →
  Disk image location 改。
- 国内直连 Docker Hub 拉不到镜像时，从镜像站拉完重新打标签即可：

```powershell
docker pull docker.m.daocloud.io/library/redis:8-alpine
docker tag docker.m.daocloud.io/library/redis:8-alpine redis:8-alpine
```

起来之后把地址填进 `.env`：`REDIS_URL=redis://:你的密码@localhost:6379/0`。
没配或连不上时，后端会在启动阶段（`init_db()` 自检）直接报错，而不是等第一个请求进来才失败。

除会话与记忆外，其余三块仍是可选的降级行为：

- 缓存：[cache.py] 的 `get_client()` 在拿不到连接时返回 `None`，
  所有缓存读写退化成空操作，链路照常跑，只是每次都重新算。
- 限流：[ratelimit.py]在拿不到连接时直接放行；
  也可以用 `RATE_LIMIT_ENABLED=false` 显式关掉。
- 任务队列：[tasks.py]的 `submit()` 返回 `None` 时，
  `/api/upload` 退回同步处理，前端直接拿到答案（不再需要 worker 进程）。

> 换嵌入模型（`EMBEDDING_MODEL`）后，RediSearch 索引的维度会与新向量不匹配，需要
> `FT.DROPINDEX {prefix}:memidx:episodic_memory` 与 `...:user_profile` 删掉索引、
> 清理对应的 `{prefix}:memdoc:*` 后重新积累记忆；知识库的 Chroma 集合也要重建。

### 5. 启动后端

```bash
python main.py
# 或： uvicorn main:app --host localhost --port 8000 --reload
```

后端默认运行在 `http://localhost:8000`。

### 6. 启动上传任务 worker（可选，配了 Redis 才需要）

```bash
python worker.py           # 前台常驻，Ctrl+C 退出
python worker.py --once    # 只处理当前队列里的任务，处理完退出（调试用）
```

不启动它的话：`/api/upload` 照样返回 `202` + `task_id`，但任务会一直停在 `queued`
（文件留在 `uploaded_files/` 里等 worker 来取），前端会一直显示「排队中...」。
所以**只要配了 Redis，就得把 worker 跑起来**。

### 7. 启动前端

```bash
cd music-chatbot-frontend
npm install
npm start
```

前端默认运行在 `http://localhost:3000`，接口地址在
[api.js](file:///d:/bishe/music-chatbot-frontend/src/service/api.js#L4-L9) 中配置（`baseURL: http://localhost:8000`）。

---

## 评测

### RGB 中文基准（zh_refine.json）

在 RGB 中文数据集的 `zh_refine.json`（300 条）上跑了完整 RAG 链路，**准确率 91%**（273/300）：

| 指标 | 值 |
| --- | --- |
| 样本数 | 300 |
| 命中率（RGB 官方 `all_rate`） | **0.91**（273 条） |
| 拒答率（`refusal_rate`） | 0.00 |
| 异常条数 | 0 |
| 单条平均耗时 | 20.4 s |

评测口径与 RGB 官方一致：每条样本把自带的 `positive` + `negative` 合并成一篇文本，
按固定长度切分（`--chunker fixed`，`chunk_size=300`）切成检索单元当知识库，
只把 `query` 交给 `rag.get_result_evaluate()`——通道规划、集合路由、网络搜索被关闭，
「查询重写 → 混合检索 → 重排 → 补充检索 → 自反思 → 生成」这条链路与线上完全一致。
判定用官方 `checkanswer`：参考答案的每一条都要作为子串出现在生成答案中才算命中。

```bash
python rgb_eval.py --chunker fixed              # 全量 300 条，结果落在 eval_results/
python rgb_eval.py --chunker fixed --limit 20   # 先跑 20 条试水
```

`--chunker` 可在 `fixed`（固定长度，配 `--overlap` 设重叠）与 `semantic`（dynamic_chunk 的
语义边界切分）之间切换，两种切法的结果分文件存放，便于对比切分方式的影响。逐条明细
（生成答案、检索上下文、命中标签）落在 `eval_results/rgb_zh_{chunker}_predictions.jsonl`，
汇总落在同目录的 `_summary.json`。

### 单条质量评测（llm_ev.py）

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
| `cache.ready` | Redis 连接建立成功（密码已抹掉） | `url` |
| `cache.disabled` / `cache.unavailable` | 未配 `REDIS_URL` / 连不上 Redis，缓存降级为直算（只打一次） | `detail` |
| `cache.error` | 运行期缓存读写异常，已忽略并继续（只打一次） | `detail` |
| `answer.cached` | 最终答案命中缓存，整条链路短路 | `question`、`chars` |
| `ratelimit.rejected` | 令牌桶耗尽，请求被拒（返回 429） | `bucket`、`identity`、`retry_after`、`capacity` |
| `ratelimit.error` | 限流组件自身异常，已放行 | `bucket`、`error` |
| `task.queued` | 上传任务入队成功 | `task_id`、`file`、`bytes`、`has_question` |
| `task.done` | 上传任务处理完成 | `task_id`、`chars` |
| `task.failed` / `task.unsupported` | 上传任务失败（后者是可预期的格式不支持），原因已写进任务记录 | `task_id`、`error` |
| `task.missing` | worker 取到任务但记录已过期（队列里有 id、hash 却没了） | `task_id` |
| `task.submit_failed` / `task.load_failed` / `task.update_failed` / `task.claim_failed` / `task.spool_cleanup_failed` | 队列自身读写异常，已忽略或退回同步 | `task_id`、`error`、`path` |
| `worker.start` / `worker.no_redis` | worker 启动参数 / 拿不到 Redis 连接（此时上传走同步，队列用不上） | `once`、`detail` |

**已埋点的步骤**：`request`（请求级）→ `rag.total`（RAG 链路汇总）→
`retrieval_planner`、`query_rewriting`、`collection_router`、`vector_search`、`rerank`、
`web_search`、`web_rewriting`、`retrieval_sufficiency`、`self_reflection`、`generate`。

其中 `query_rewriting` / `collection_router` / `vector_search` 会多带一个 `cached` 字段
（`true` 表示这一步直接读了 Redis，耗时接近于 0），离线看链路时一眼能分辨
「这次快是因为缓存」还是「真的算得快」。

补充检索会让同一 `request_id` 下出现多个同名的 `vector_search` / `web_search` / `query_rewriting` span，
用 `span_id` 区分轮次（与「多个集合各检索一次」是同一套机制）。通道被判定为不需要时，
对应的 `collection_router` / `web_rewriting` 等 span 会**整条不出现**，可据此确认通道确实没被白跑。

上传任务由 worker 进程执行，除了上面这些步骤还会多一个 `task.upload` span 作为根节点
（`document.parse`、`memory.build_context`、`rag` 各步都挂在它下面）。因为入队时记了
`request_id`，用 `python trace_view.py app.log --request-id <原请求id>` 仍能把这棵树完整还原。

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
