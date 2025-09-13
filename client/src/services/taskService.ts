import { Task, CreateTaskInput, UpdateTaskInput } from '@/types/task';
import { supabase } from '@/integrations/supabase/client';

// API base URL - adjust this to match your backend URL
const API_BASE_URL = '/api';

export const taskService = {
  async getAllTasks(): Promise<Task[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Include cookies for authentication
      });
      
      if (!response.ok) {
        throw new Error(`Error fetching tasks: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.tasks;
    } catch (error) {
      console.error('Failed to fetch tasks:', error);
      throw error;
    }
  },

  async getTaskById(id: string): Promise<Task | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (response.status === 404) {
        return null;
      }
      
      if (!response.ok) {
        throw new Error(`Error fetching task: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Failed to fetch task ${id}:`, error);
      throw error;
    }
  },

  async createTask(taskInput: CreateTaskInput): Promise<Task> {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/run-task`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(taskInput),
      });
      
      if (!response.ok) {
        throw new Error(`Error creating task: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to create task:', error);
      throw error;
    }
  },

  async updateTask(id: string, updates: UpdateTaskInput): Promise<Task> {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(updates),
      });
      
      if (!response.ok) {
        throw new Error(`Error updating task: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Failed to update task ${id}:`, error);
      throw error;
    }
  },

  async deleteTask(id: string): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error(`Error deleting task: ${response.statusText}`);
      }
    } catch (error) {
      console.error(`Failed to delete task ${id}:`, error);
      throw error;
    }
  },

  async getTasksByStatus(status: string): Promise<Task[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/by-status/${status}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error(`Error fetching tasks by status: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.tasks;
    } catch (error) {
      console.error(`Failed to fetch tasks with status ${status}:`, error);
      throw error;
    }
  }
};
