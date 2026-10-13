import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { chatAPI } from './service/api.js';
import './App.css';

// 登录态：账号 → user_id 的映射由后端 auth.py 维护，前端只保存登录结果
// （user_id + 账号名，不存密码）。刷新页面仍是同一个人，会话与跨会话记忆才连得上。
const USER_ID_KEY = 'music_rag_user_id';
const USERNAME_KEY = 'music_rag_username';

function readIdentity() {
  try {
    return {
      userId: localStorage.getItem(USER_ID_KEY) || '',
      username: localStorage.getItem(USERNAME_KEY) || '',
    };
  } catch (error) {
    // localStorage 不可用（隐私模式等）时按未登录处理，用户当场登录即可
    return { userId: '', username: '' };
  }
}

function persistIdentity(userId, username) {
  try {
    localStorage.setItem(USER_ID_KEY, userId);
    localStorage.setItem(USERNAME_KEY, username);
  } catch (error) {
    // 存不下也不阻断本次使用，只是刷新后需要重新登录
  }
}

function clearIdentity() {
  try {
    localStorage.removeItem(USER_ID_KEY);
    localStorage.removeItem(USERNAME_KEY);
  } catch (error) {
    // 忽略：清不掉也能通过页面上的退出按钮回到登录页
  }
}

// 登录页：登录 / 注册共用一张表单，成功后把 { user_id, username } 交给 App。
function LoginPage({ onAuthenticated }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const isRegister = mode === 'register';

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;
    setError('');
    setSubmitting(true);
    try {
      const result = isRegister
        ? await chatAPI.register(username, password)
        : await chatAPI.login(username, password);
      onAuthenticated(result.user_id, result.username);
    } catch (err) {
      // 后端的报错（账号已存在 / 账号或密码不正确 / 用户名格式…）原样展示，别自己改写
      const detail = err?.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : '请求失败，请稍后重试');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={handleSubmit}>
        <h1>
          <span className="brand-mark" aria-hidden="true">♪</span>
          乐典音乐助手
        </h1>
        <p className="login-subtitle">
          {isRegister ? '注册一个新账号' : '登录后继续你的对话'}
        </p>

        <label className="login-label" htmlFor="login-username">账号</label>
        <input
          id="login-username"
          className="login-input"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="3-32 位字母、数字、下划线、点或连字符"
          autoComplete="username"
          required
        />

        <label className="login-label" htmlFor="login-password">密码</label>
        <input
          id="login-password"
          className="login-input"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder={isRegister ? '至少 6 位' : '请输入密码'}
          autoComplete={isRegister ? 'new-password' : 'current-password'}
          required
        />

        {error && <p className="login-error">{error}</p>}

        <button type="submit" className="btn btn-primary login-submit" disabled={submitting}>
          {submitting ? '处理中…' : isRegister ? '注册并登录' : '登录'}
        </button>

        <button
          type="button"
          className="login-switch"
          onClick={() => {
            setMode(isRegister ? 'login' : 'register');
            setError('');
          }}
        >
          {isRegister ? '已有账号？去登录' : '没有账号？去注册'}
        </button>
      </form>
    </div>
  );
}

// 会话列表里每条记录的时间显示（后端给的是 ISO 字符串，解析失败就留空）
function formatSessionTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString([], {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [sessions, setSessions] = useState([]);
  // 登录态：必须有 user_id 和账号名才算登录（两个一起写、一起清）。
  // 只认到旧版的匿名 user_id（改造前随机生成、没有账号名）也当作未登录，重新登录一次，
  // 否则旧身份会绕过登录页，用户会以为登录界面没生效。
  const [identity, setIdentity] = useState(readIdentity);
  const userId = identity.userId;
  const username = identity.username;
  const loggedIn = Boolean(userId && username);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStage, setUploadStage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileQuestion, setFileQuestion] = useState('');
  const [selfIntroduction, setSelfIntroduction] = useState('');
  const [musicAnalysis, setMusicAnalysis] = useState('');
  const [musicList, setMusicList] = useState('');
  const [showKnowledge, setShowKnowledge] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 会话列表：侧边栏导航用。拉取失败就退化成空列表，不影响问答主链路。
  const refreshSessions = useCallback(async () => {
    if (!userId) {
      setSessions([]);
      return;
    }
    try {
      const data = await chatAPI.listSessions(userId);
      setSessions(data.sessions || []);
    } catch (error) {
      setSessions([]);
    }
  }, [userId]);

  // 进页面（或换账号登录后）就列出该账号名下的历史会话
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  // 登录 / 注册成功：记下身份，主界面随即渲染并拉取该账号的会话列表
  const handleAuthenticated = (newUserId, newUsername) => {
    persistIdentity(newUserId, newUsername);
    setIdentity({ userId: newUserId, username: newUsername });
  };

  // 退出登录：清掉本地身份并复位界面，不删服务端任何数据
  const handleLogout = () => {
    clearIdentity();
    setIdentity({ userId: '', username: '' });
    setSessionId('');
    setMessages([]);
    setSessions([]);
  };

  // 处理发送消息
  const handleSendMessage = async () => {
    if (!inputText.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: inputText,
      timestamp: new Date().toISOString(),
    };

    // 添加到消息列表
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);

    try {
      // 发送到后端，sessionId为null时由后端生成新的session_id
      const response = await chatAPI.sendMessage({
        question: inputText,
        sessionId: sessionId, // 直接使用当前的sessionId，如果为null则由后端生成
        userId, // 身份，后端据此隔离跨会话记忆
      });

      // 更新会话ID（后端会返回正确的session_id）
      setSessionId(response.session_id);

      // 添加AI回复到消息列表
      const aiMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: response.timestamp,
      };
      setMessages(prev => [...prev, aiMessage]);
      // 新一轮问答后刷新侧边栏：新会话要出现，已有会话要按最近活跃排到最前
      refreshSessions();
    } catch (error) {
      // 显示错误消息
      const errorMessage = {
        role: 'assistant',
        content: '抱歉，处理请求时出现错误。请稍后重试。',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // 处理回车键发送
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 把某段会话的历史载入主区域（后端会把较早轮次压缩成摘要，一并展示，
  // 避免"加载历史后只剩最后几条"）
  const loadSessionHistory = async (id) => {
    const response = await chatAPI.getHistory(id);
    const historyMessages = response.messages || [];
    const summaryMessage = response.summary
      ? [{
          role: 'assistant',
          content: `【此前对话摘要】\n${response.summary}`,
          timestamp: response.timestamp,
        }]
      : [];
    setMessages([...summaryMessage, ...historyMessages]);
  };

  // 点击侧边栏一条会话：切到该会话并载入它的历史；之后继续发送就接在这段对话上
  const handleOpenSession = async (id) => {
    if (id === sessionId) return;
    setSessionId(id);
    try {
      await loadSessionHistory(id);
    } catch (error) {
      alert('加载历史记录失败');
    }
  };

  // 清空历史记录
  const handleClearHistory = async () => {
    if (!sessionId) {
      alert('当前没有会话');
      return;
    }

    if (window.confirm('确定要清空当前会话的历史记录吗？')) {
      try {
        await chatAPI.deleteHistory(sessionId);
        setMessages([]);
        alert('历史记录已清空');
      } catch (error) {
        alert('清空历史记录失败');
      }
    }
  };

  // 新会话
  const handleNewSession = () => {
    // 清空当前会话状态，后端会在下次请求时生成新的session_id
    setSessionId(null);
    setMessages([]);
  };

  // 处理文件选择
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  // 处理文件上传
  // 解析文档要逐张调视觉模型，几十张图的 PDF 可能跑几分钟，所以后端是异步的：
  // 提交后立刻返回 task_id，这边轮询 /api/task/{id} 直到 done / failed。
  // 没配 Redis 时后端退回同步处理，返回的 status 直接是 done，就不进轮询了。
  const UPLOAD_POLL_INTERVAL_MS = 1500;
  const UPLOAD_POLL_TIMEOUT_MS = 10 * 60 * 1000;
  const UPLOAD_STAGE_TEXT = { queued: '排队中...', running: '解析中...' };

  const waitForUploadTask = async (taskId) => {
    const deadline = Date.now() + UPLOAD_POLL_TIMEOUT_MS;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, UPLOAD_POLL_INTERVAL_MS));
      const task = await chatAPI.getTask(taskId);
      if (task.status === 'done') return task;
      if (task.status === 'failed') throw new Error(task.error || '文件处理失败');
      setUploadStage(UPLOAD_STAGE_TEXT[task.status] || '处理中...');
    }
    throw new Error('处理超时，请刷新页面后到历史记录里查看');
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      alert('请先选择文件');
      return;
    }

    let userContent = `已上传文件: ${selectedFile.name}`;
    if (fileQuestion.trim()) {
      userContent += `\n问题: ${fileQuestion.trim()}`;
    }

    setUploading(true);
    setUploadStage('上传中...');
    // 用户消息先上屏：解析要跑几分钟，等结束再显示会让人以为没点上
    setMessages(prev => [...prev, {
      role: 'user',
      content: userContent,
      timestamp: new Date().toISOString(),
    }]);

    try {
      const submitted = await chatAPI.uploadFile(
        selectedFile, fileQuestion.trim() || null, sessionId, userId
      );
      const result = submitted.status === 'done'
        ? submitted
        : await waitForUploadTask(submitted.task_id);

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.answer,
        timestamp: result.timestamp,
      }]);

      // 更新会话ID（后端会返回正确的session_id）
      setSessionId(result.session_id);
      // 上传问答同样会新建/更新会话，侧边栏跟着刷新
      refreshSessions();

      setSelectedFile(null);
      setFileQuestion('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      // 后端 4xx 的原因写在 detail 里（比如文件格式不支持），比笼统的"重试"有用
      const detail = error?.response?.data?.detail || error?.message || '请稍后重试';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `文件处理失败：${detail}`,
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setUploading(false);
      setUploadStage('');
    }
  };

  // 处理添加个人信息
  const handleAddSelfIntroduction = async () => {
    if (!selfIntroduction.trim()) {
      alert('请输入个人信息');
      return;
    }

    try {
      await chatAPI.addSelfIntroduction(selfIntroduction);
      alert('个人信息添加成功');
      setSelfIntroduction('');
    } catch (error) {
      alert('添加失败，请重试');
    }
  };

  // 处理添加音乐理解
  const handleAddMusicAnalysis = async () => {
    if (!musicAnalysis.trim()) {
      alert('请输入音乐理解');
      return;
    }

    try {
      await chatAPI.addMusicAnalysis(musicAnalysis);
      alert('音乐理解添加成功');
      setMusicAnalysis('');
    } catch (error) {
      alert('添加失败，请重试');
    }
  };

  // 处理添加歌单
  const handleAddMusicList = async () => {
    if (!musicList.trim()) {
      alert('请输入歌单');
      return;
    }

    try {
      await chatAPI.addMusicList(musicList);
      alert('歌单添加成功');
      setMusicList('');
    } catch (error) {
      alert('添加失败，请重试');
    }
  };

  // 未登录：只渲染登录页。会话、记忆都挂在账号上，没身份就没有可展示的内容。
  if (!loggedIn) {
    return <LoginPage onAuthenticated={handleAuthenticated} />;
  }

  return (
    <div className="app">
      <header className="header">
        <h1>
          <span className="brand-mark" aria-hidden="true">♪</span>
          乐典音乐助手
        </h1>
        <p className="subtitle">专注音乐领域的知识问答系统</p>
        <div className="header-user">
          <span className="header-username" title={userId}>{username}</span>
          <button type="button" className="btn btn-secondary btn-logout" onClick={handleLogout}>
            退出登录
          </button>
        </div>
      </header>

      <div className="main-container">
        {/* 侧边栏 - 会话管理 */}
        <aside className="sidebar">
          <div className="session-section">
            <h3>会话管理</h3>
            <div className="session-controls">
              <button 
                onClick={handleNewSession}
                className="btn btn-primary"
              >
                新会话
              </button>
              <button 
                onClick={handleClearHistory}
                className="btn btn-danger"
              >
                清空历史
              </button>
            </div>

            <div className="session-list-section">
              <h4>历史会话</h4>
              {sessions.length === 0 ? (
                <p className="session-hint">
                  还没有历史会话，发送第一条消息后会出现在这里。
                </p>
              ) : (
                <ul className="session-list">
                  {sessions.map((item) => (
                    <li key={item.session_id}>
                      <button
                        type="button"
                        className={
                          'session-list-item' +
                          (item.session_id === sessionId ? ' active' : '')
                        }
                        onClick={() => handleOpenSession(item.session_id)}
                        title={item.session_id}
                      >
                        <span className="session-item-title">
                          {item.title || '未命名会话'}
                        </span>
                        <span className="session-item-time">
                          {formatSessionTime(item.updated_at || item.created_at)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <h4>身份 ID</h4>
              <p className="session-hint">{userId}</p>
              <p className="session-hint">跨会话记忆按该身份隔离，清除浏览器数据会重新生成</p>
            </div>

            <div className="knowledge-section">
              <button
                type="button"
                className="section-toggle"
                onClick={() => setShowKnowledge(prev => !prev)}
                aria-expanded={showKnowledge}
                aria-controls="knowledge-panel"
              >
                <span>知识库管理</span>
                <span className="toggle-chevron" aria-hidden="true">▾</span>
              </button>

              {showKnowledge && (
              <div className="knowledge-panel" id="knowledge-panel">
              <p className="section-hint">
                添加的内容会写入知识库，供助手在回答时检索。
              </p>

              <div className="knowledge-item">
                <h5>个人信息</h5>
                <textarea
                  value={selfIntroduction}
                  onChange={(e) => setSelfIntroduction(e.target.value)}
                  placeholder="输入你的个人信息..."
                  className="knowledge-textarea"
                  rows="3"
                />
                <button
                  onClick={handleAddSelfIntroduction}
                  className="btn btn-primary btn-small"
                >
                  添加
                </button>
              </div>

              <div className="knowledge-item">
                <h5>音乐理解</h5>
                <textarea
                  value={musicAnalysis}
                  onChange={(e) => setMusicAnalysis(e.target.value)}
                  placeholder="输入你对音乐的理解..."
                  className="knowledge-textarea"
                  rows="3"
                />
                <button
                  onClick={handleAddMusicAnalysis}
                  className="btn btn-primary btn-small"
                >
                  添加
                </button>
              </div>

              <div className="knowledge-item">
                <h5>歌单</h5>
                <textarea
                  value={musicList}
                  onChange={(e) => setMusicList(e.target.value)}
                  placeholder="输入你喜欢的歌单..."
                  className="knowledge-textarea"
                  rows="3"
                />
                <button
                  onClick={handleAddMusicList}
                  className="btn btn-primary btn-small"
                >
                  添加
                </button>
              </div>
              </div>
              )}
            </div>
          </div>
        </aside>

        {/* 主聊天区域 */}
        <main className="chat-container">
          {/* 消息列表 */}
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🎵</div>
                <h3>欢迎使用乐典音乐助手！</h3>
                <p>我是一个专业的音乐知识问答助手，可以回答你关于音乐的各种问题。</p>
                <p>例如：</p>
                <div className="example-questions">
                  {[
                    '贝多芬的第五交响曲有什么特点？',
                    '什么是蓝调音乐？',
                    '推荐一些古典音乐作品',
                  ].map((question) => (
                    <button
                      key={question}
                      type="button"
                      className="example-question"
                      onClick={() => {
                        setInputText(question);
                        inputRef.current?.focus();
                      }}
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`message ${message.role === 'user' ? 'user-message' : 'ai-message'}`}
                >
                  <div className="message-avatar" aria-hidden="true">
                    {message.role === 'user' ? '你' : '乐'}
                  </div>
                  <div className="message-content">
                    <div className="message-role">
                      {message.role === 'user' ? '你' : '乐典助手'}
                      {message.timestamp && (
                        <span className="message-time">
                          {new Date(message.timestamp).toLocaleTimeString([], { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                        </span>
                      )}
                    </div>
                    {/* 助手回复按 Markdown 渲染，用户自己的输入保持纯文本原样显示 */}
                    <div className="message-text">
                      {message.role === 'user' ? (
                        message.content
                      ) : (
                        <div className="markdown">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="message ai-message">
                <div className="message-avatar" aria-hidden="true">乐</div>
                <div className="message-content">
                  <div className="message-role">乐典助手</div>
                  <div className="message-text typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* 输入区域 */}
          <div className="input-container">
            <div className="input-wrapper">
              <div className="file-upload-section">
                <div className="file-upload-row">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    className="file-input"
                    accept=".pdf,.docx,.png,.jpg,.jpeg,.gif,.webp"
                  />
                  {selectedFile && (
                    <div className="selected-file">
                      <span>已选择: {selectedFile.name}</span>
                      <button
                        onClick={() => {
                          setSelectedFile(null);
                          setFileQuestion('');
                          if (fileInputRef.current) {
                            fileInputRef.current.value = '';
                          }
                        }}
                        className="clear-file-btn"
                      >
                        ✕
                      </button>
                    </div>
                  )}
                </div>
                {selectedFile && (
                  <input
                    type="text"
                    value={fileQuestion}
                    onChange={(e) => setFileQuestion(e.target.value)}
                    placeholder="输入关于文件的问题（可选）"
                    className="file-question-input"
                  />
                )}
                <button
                  onClick={handleFileUpload}
                  disabled={!selectedFile || uploading}
                  className="upload-button"
                >
                  {uploading ? (uploadStage || '上传中...') : '上传文件'}
                </button>
              </div>
              <textarea
                ref={inputRef}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="输入你的问题（按 Enter 发送，Shift+Enter 换行）..."
                rows="3"
                disabled={loading}
                className="message-input"
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputText.trim() || loading}
                className="send-button"
              >
                {loading ? (
                  <span className="spinner"></span>
                ) : (
                  '发送'
                )}
              </button>
            </div>
            <div className="input-hint">
              检索方式由助手依据提问自动决定（知识库 / 网络搜索 / 两者）
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
