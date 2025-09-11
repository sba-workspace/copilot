export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed';

export interface Task {
  id: string;
  description: string;
  status: TaskStatus;
  user_id: string;
  created_at: string | null;
  updated_at: string | null;
  metadata: any | null;
  agent_type: string | null;
  result: string | null;
}

export interface CreateTaskInput {
  description: string;
}

export interface UpdateTaskInput {
  description?: string;
  status?: TaskStatus;
  metadata?: any | null;
  result?: string | null;
}
