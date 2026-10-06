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
  // 发送消息
  // 是否走知识库/网络检索由后端模型依据提问决定，前端不再传开关
  // userId：用户身份，后端按它隔离情景记忆与用户画像（跨会话记忆的前提）
  async sendMessage({ question, sessionId, userId }) {
    try {
      const response = await api.post('/api/chat', {
        question,
        session_id: sessionId,
        user_id: userId,
      });
      return response.data;
    } catch (error) {
      console.error('发送消息失败:', error);
      throw error;
    }
  },

  // 获取历史记录
  async getHistory(sessionId, number = 20) {
    try {
      const response = await api.get(`/api/chat/history/${sessionId}`, {
        params: { number }
      });
      return response.data;
    } catch (error) {
      console.error('获取历史记录失败:', error);
      throw error;
    }
  },

  // 删除历史记录
  async deleteHistory(sessionId) {
    try {
      const response = await api.delete(`/api/chat/history/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('删除历史记录失败:', error);
      throw error;
    }
  },

  // 上传文件
  // 传 sessionId / userId：上传也要落在同一个会话与同一个身份下，
  // 否则每上传一次都会新建会话，情景记忆的归属也会出错
  // 返回两种形态：{status:'queued', task_id}（解析在 worker 里跑，需轮询 getTask）
  //              {status:'done', answer, session_id, timestamp}（后端没 Redis，同步兜底直接给答案）
  async uploadFile(file, question = null, sessionId = null, userId = null) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (question) {
        formData.append('question', question);
      }
      if (sessionId) {
        formData.append('session_id', sessionId);
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