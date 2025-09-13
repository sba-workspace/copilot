import { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useQueryClient } from '@tanstack/react-query';

// Connection status types
export type WebSocketStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

// Task update from the WebSocket
export interface TaskUpdate {
  task_id: string;
  description: string;
  status: string;
  result: string | null;
  created_at: string | null;
  updated_at: string | null;
}

// WebSocket message types
interface TasksListMessage {
  type: 'tasks_list';
  data: TaskUpdate[];
}

interface TaskUpdateMessage {
  type: 'task_update';
  data: TaskUpdate;
}

type WebSocketMessage = TasksListMessage | TaskUpdateMessage;

// Define the context type
interface WebSocketContextType {
  status: WebSocketStatus;
  tasks: TaskUpdate[];
  refreshTasks: () => void;
  lastError: string | null;
}

// Create context with default values
const WebSocketContext = createContext<WebSocketContextType>({
  status: 'idle',
  tasks: [],
  refreshTasks: () => {},
  lastError: null
});

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { session } = useAuth();
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<WebSocketStatus>('idle');
  const [tasks, setTasks] = useState<TaskUpdate[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const maxReconnectDelay = 5000;
  const reconnectAttempts = useRef(0);

  // Function to establish WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (!session?.access_token) return;
    
    try {
      setStatus('connecting');
      
      // Close existing connection if any
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }

      // Create new WebSocket connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/realtime/tasks-feed?token=${session.access_token}`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // WebSocket event handlers
      ws.onopen = () => {
        setStatus('connected');
        setLastError(null);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          
          if (message.type === 'tasks_list') {
            // Update local state
            setTasks(message.data);
            
            // Update TanStack Query cache
            queryClient.setQueryData(['tasks'], message.data);
          } else if (message.type === 'task_update') {
            // Update local state
            setTasks(prev => {
              // Find if task already exists and update it, or add new task
              const exists = prev.some(task => task.task_id === message.data.task_id);
              if (exists) {
                return prev.map(task => 
                  task.task_id === message.data.task_id ? message.data : task
                );
              } else {
                return [...prev, message.data];
              }
            });
            
            // Update specific task in TanStack Query cache
            queryClient.setQueryData(['tasks'], (oldData: TaskUpdate[] | undefined) => {
              if (!oldData) return [message.data];
              
              const exists = oldData.some(task => task.task_id === message.data.task_id);
              if (exists) {
                return oldData.map(task => 
                  task.task_id === message.data.task_id ? message.data : task
                );
              } else {
                return [...oldData, message.data];
              }
            });
            
            // Also update the individual task cache if it exists
            queryClient.setQueryData(['task', message.data.task_id], message.data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('error');
        setLastError('Connection error. Attempting to reconnect...');
      };

      ws.onclose = (event) => {
        setStatus('disconnected');
        // Attempt to reconnect with exponential backoff
        if (!event.wasClean) {
          const delay = Math.min(1000 * (2 ** reconnectAttempts.current), maxReconnectDelay);
          reconnectAttempts.current += 1;
          
          console.log(`WebSocket closed. Reconnecting in ${delay}ms...`);
          
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (session?.access_token) {
              connectWebSocket();
            }
          }, delay);
        }
      };
    } catch (error) {
      console.error('Error establishing WebSocket connection:', error);
      setStatus('error');
      setLastError('Failed to establish connection');
    }
  }, [session?.access_token, queryClient]);

  // Function to manually refresh tasks
  const refreshTasks = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send('refresh');
      
      // Also invalidate the tasks query to ensure fresh data
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    }
  }, [queryClient]);

  // Connect WebSocket when session changes
  useEffect(() => {
    if (session?.access_token) {
      connectWebSocket();
    } else {
      setStatus('idle');
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    }

    // Clean up function
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [session, connectWebSocket]);

  return (
    <WebSocketContext.Provider value={{ status, tasks, refreshTasks, lastError }}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Custom hook for using the WebSocket context
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}; 