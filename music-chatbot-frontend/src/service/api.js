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
  async sendMessage({ question, sessionId, useWebSearch = true, useKnowledgeBase = true }) {
    try {
      const response = await api.post('/api/chat', {
        question,
        session_id: sessionId,
        use_web_search: useWebSearch,
        use_knowledge_base: useKnowledgeBase,
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
  async uploadFile(file, question = null) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (question) {
        formData.append('question', question);
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