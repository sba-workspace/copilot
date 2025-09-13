import { v4 as uuidv4 } from 'uuid';

// API base URL - adjust this to match your backend URL
const API_BASE_URL = '/api';

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
}

export const chatService = {
  async sendMessage(message: string, conversationId?: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Include cookies for authentication
        body: JSON.stringify({
          message,
          conversation_id: conversationId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error sending message: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  },

  // Helper function to format messages for the UI
  formatUserMessage(content: string): ChatMessage {
    return {
      id: uuidv4(),
      content,
      sender: 'user',
      timestamp: new Date()
    };
  },

  formatAssistantMessage(content: string): ChatMessage {
    return {
      id: uuidv4(),
      content,
      sender: 'assistant',
      timestamp: new Date()
    };
  }
}; 