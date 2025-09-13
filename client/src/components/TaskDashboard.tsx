import * as React from 'react';
import { useState } from 'react';
import { Plus, Calendar } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { taskService } from '@/services/taskService';
import { Task, TaskStatus } from '@/types/task';
import { toast } from '@/hooks/use-toast';
import { AddTaskModal } from './AddTaskModal';

export const TaskDashboard = () => {
  const queryClient = useQueryClient();
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  
  // Get all tasks
  const { data: tasks = [], isLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: taskService.getAllTasks
  });

  // Update task mutation
  const updateTaskMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: any }) => 
      taskService.updateTask(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      toast({
        title: "Task updated",
        description: "The task has been updated successfully."
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error updating task",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const columns = [
    { id: 'pending', title: 'Pending', color: 'border-gray-500' },
    { id: 'in_progress', title: 'In Progress', color: 'border-blue-500' },
    { id: 'completed', title: 'Completed', color: 'border-green-500' },
    { id: 'failed', title: 'Failed', color: 'border-red-500' }
  ];

  const getTasksByStatus = (status: string) => {
    return tasks.filter(task => task.status === status);
  };

  const handleDragStart = (e: React.DragEvent, taskId: string) => {
    e.dataTransfer.setData('taskId', taskId);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, status: TaskStatus) => {
    e.preventDefault();
    const taskId = e.dataTransfer.getData('taskId');
    const task = tasks.find(t => t.id === taskId);
    
    if (task && task.status !== status) {
      updateTaskMutation.mutate({
        id: taskId,
        updates: { status }
      });
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-700">
        <div>
          <h1 className="text-2xl font-semibold text-white">Task Dashboard</h1>
          <p className="text-gray-400">Manage your tasks and workflows</p>
        </div>
        <button 
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
          onClick={() => setIsAddModalOpen(true)}
        >
          <Plus className="w-4 h-4" />
          <span>Add Task</span>
        </button>
      </div>

      {/* Kanban Board */}
      <div className="flex-1 p-6 overflow-x-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : (
          <div className="flex space-x-6 h-full min-w-max">
            {columns.map((column) => (
              <div 
                key={column.id} 
                className="flex-1 min-w-80"
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(e, column.id as TaskStatus)}
              >
                <div className={cn(
                  "rounded-lg border-2 border-dashed p-4 h-full",
                  column.color
                )}>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold text-white">{column.title}</h2>
                    <span className="bg-gray-700 text-gray-300 px-2 py-1 rounded-full text-sm">
                      {getTasksByStatus(column.id).length}
                    </span>
                  </div>
                  
                  <div className="space-y-4">
                    {getTasksByStatus(column.id).map((task) => (
                      <div
                        key={task.id}
                        className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:bg-gray-750 transition-colors cursor-pointer"
                        draggable
                        onDragStart={(e) => handleDragStart(e, task.id)}
                      >
                        <p className="text-white font-medium mb-2">{task.description}</p>
                        
                        <div className="flex items-center justify-between text-xs text-gray-400">
                          {task.created_at && (
                            <div className="flex items-center space-x-1">
                              <Calendar className="w-3 h-3" />
                              <span>{new Date(task.created_at).toLocaleDateString()}</span>
                            </div>
                          )}
                          {task.agent_type && (
                            <div className="bg-blue-900 text-blue-200 px-2 py-1 rounded text-xs">
                              {task.agent_type}
                            </div>
                          )}
                        </div>
                        
                        {task.result && (
                          <div className="mt-2 text-sm text-gray-400 border-t border-gray-700 pt-2">
                            <p className="line-clamp-2">{task.result}</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Add Task Modal */}
      <AddTaskModal isOpen={isAddModalOpen} onClose={() => setIsAddModalOpen(false)} />
    </div>
  );
};
