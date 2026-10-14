import axios from 'axios';

// 创建 axios 实例
const api = axios.create({
  baseURL: 'http://localhost:8000', 
  headers: {
    'Content-Type': 'application/json',
  },
});

// API 调用函数
export const chatAPI = {
  // 注册：成功返回 { user_id, username }；账号已存在时后端回 409
  async register(username, password) {
    try {
      const response = await api.post('/api/auth/register', { username, password });
      return response.data;
    } catch (error) {
      console.error('注册失败:', error);
      throw error;
    }
  },

  // 登录：成功返回 { user_id, username }；账号或密码不对时后端回 401
  async login(username, password) {
    try {
      const response = await api.post('/api/auth/login', { username, password });
      return response.data;
    } catch (error) {
      console.error('登录失败:', error);
      throw error;
    }
  },

  // 发送消息
  // 是否走知识库/网络检索由后端模型依据提问决定，前端不再传开关
  // userId：用户身份，后端按它隔离情景记忆与用户画像，也是「一个账号一个会话」的归属键
  async sendMessage({ question, userId }) {
    try {
      const response = await api.post('/api/chat', {
        question,
        user_id: userId,
      });
      return response.data;
    } catch (error) {
      console.error('发送消息失败:', error);
      throw error;
    }
  },

  // 获取历史记录：按 user_id 取该账号唯一会话的最近消息
  // 单会话下所有对话都堆在同一个会话里，默认多取一些，避免登录后只能看到最近几条
  async getHistory(userId, number = 100) {
    try {
      const response = await api.get('/api/chat/history', {
        params: { user_id: userId, number }
      });
      return response.data;
    } catch (error) {
      console.error('获取历史记录失败:', error);
      throw error;
    }
  },

  // 上传文件
  // 只传 userId：上传也落在同一个账号唯一的会话下；不再需要 session_id
  // 返回两种形态：{status:'queued', task_id}（解析在 worker 里跑，需轮询 getTask）
  //              {status:'done', answer, session_id, timestamp}（后端没 Redis，同步兜底直接给答案）
  async uploadFile(file, question = null, userId = null) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (question) {
        formData.append('question', question);
      }
      if (userId) {
        formData.append('user_id', userId);
      }
      
      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('上传文件失败:', error);
      throw error;
    }
  },

  // 查询异步任务状态：status 为 queued / running / done / failed
  async getTask(taskId) {
    try {
      const response = await api.get(`/api/task/${taskId}`);
      return response.data;
    } catch (error) {
      console.error('查询任务状态失败:', error);
      throw error;
    }
  },

  // 添加个人信息
  async addSelfIntroduction(text) {
    try {
      const formData = new FormData();
      formData.append('text', text);
      
      const response = await api.post('/api/knowledge/self-introduction', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('添加个人信息失败:', error);
      throw error;
    }
  },

  // 添加音乐理解
  async addMusicAnalysis(text) {
    try {
      const formData = new FormData();
      formData.append('text', text);
      
      const response = await api.post('/api/knowledge/music-analysis', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('添加音乐理解失败:', error);
      throw error;
    }
  },

  // 添加歌单
  async addMusicList(text) {
    try {
      const formData = new FormData();
      formData.append('text', text);
      
      const response = await api.post('/api/knowledge/music-list', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('添加歌单失败:', error);
      throw error;
    }
  }
};