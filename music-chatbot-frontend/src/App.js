import React, { useState, useEffect, useRef } from 'react';
import { chatAPI } from './service/api.js';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileQuestion, setFileQuestion] = useState('');
  const [selfIntroduction, setSelfIntroduction] = useState('');
  const [musicAnalysis, setMusicAnalysis] = useState('');
  const [musicList, setMusicList] = useState('');
  const [settings, setSettings] = useState({
    useWebSearch: true,
    useKnowledgeBase: true,
  });
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 生成新的会话ID
  const generateNewSessionId = () => {
    // 不再在前端生成session_id，由后端统一生成
    return null;
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
        useWebSearch: settings.useWebSearch,
        useKnowledgeBase: settings.useKnowledgeBase,
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

  // 加载历史记录
  const handleLoadHistory = async () => {
    if (!sessionId) {
      alert('请输入会话ID');
      return;
    }

    try {
      const response = await chatAPI.getHistory(sessionId);
      setMessages(response.messages);
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
  const handleFileUpload = async () => {
    if (!selectedFile) {
      alert('请先选择文件');
      return;
    }

    setUploading(true);
    try {
      const response = await chatAPI.uploadFile(selectedFile, fileQuestion.trim() || null);
      
      let userContent = `已上传文件: ${selectedFile.name}`;
      if (fileQuestion.trim()) {
        userContent += `\n问题: ${fileQuestion.trim()}`;
      }
      
      const userMessage = {
        role: 'user',
        content: userContent,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, userMessage]);

      const aiMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: response.timestamp,
      };
      setMessages(prev => [...prev, aiMessage]);
      
      // 更新会话ID（后端会返回正确的session_id）
      setSessionId(response.session_id);
      
      setSelectedFile(null);
      setFileQuestion('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: '文件上传失败，请重试。',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setUploading(false);
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

  return (
    <div className="app">
      <header className="header">
        <h1>🎵 乐典音乐助手</h1>
        <p className="subtitle">-------励志于打造专业的音乐知识问答系统-------</p>
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
                onClick={handleLoadHistory}
                className="btn btn-secondary"
              >
                加载历史
              </button>
              <button 
                onClick={handleClearHistory}
                className="btn btn-danger"
              >
                清空历史
              </button>
            </div>
            
            <div className="session-info">
              <h4>会话ID</h4>
              <input
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="输入或自动生成会话ID"
                className="session-input"
              />
              <p className="session-hint">修改会话ID可切换到不同对话</p>
            </div>

            <div className="settings-section">
              <h4>设置</h4>
              <div className="setting-item">
                <label>
                  <input
                    type="checkbox"
                    checked={settings.useWebSearch}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      useWebSearch: e.target.checked
                    }))}
                  />
                  使用网络搜索
                </label>
              </div>
              <div className="setting-item">
                <label>
                  <input
                    type="checkbox"
                    checked={settings.useKnowledgeBase}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      useKnowledgeBase: e.target.checked
                    }))}
                  />
                  使用知识库
                </label>
              </div>
            </div>

            <div className="knowledge-section">
              <h4>知识库管理</h4>
              
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
                <ul className="example-questions">
                  <li>贝多芬的第五交响曲有什么特点？</li>
                  <li>什么是蓝调音乐？</li>
                  <li>推荐一些古典音乐作品</li>
                </ul>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`message ${message.role === 'user' ? 'user-message' : 'ai-message'}`}
                >
                  <div className="message-avatar">
                    {message.role === 'user' ? '👤' : '🤖'}
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
                    <div className="message-text">{message.content}</div>
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="message ai-message">
                <div className="message-avatar">🤖</div>
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
                    accept=".txt,.pdf,.doc,.docx"
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
                  {uploading ? '上传中...' : '上传文件'}
                </button>
              </div>
              <textarea
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
              当前设置: {settings.useWebSearch ? '网络搜索✅' : '网络搜索❌'} | 
              {settings.useKnowledgeBase ? '知识库✅' : '知识库❌'}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
