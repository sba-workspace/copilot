import { useState } from 'react';
import { Sidebar } from './Sidebar';
import { ChatWindow } from './ChatWindow';
import { TaskDashboard } from './TaskDashboard';
import { IntegrationsPage } from './IntegrationsPage';
import { cn } from '@/lib/utils';

export type ViewType = 'chat' | 'tasks' | 'integrations';

export const Layout = () => {
  const [currentView, setCurrentView] = useState<ViewType>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const renderMainContent = () => {
    switch (currentView) {
      case 'chat':
        return <ChatWindow />;
      case 'tasks':
        return <TaskDashboard />;
      case 'integrations':
        return <IntegrationsPage />;
      default:
        return <ChatWindow />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex w-full">
      <Sidebar 
        currentView={currentView}
        onViewChange={setCurrentView}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <main className={cn(
        "flex-1 flex flex-col transition-all duration-300",
        sidebarCollapsed ? "ml-16" : "ml-64"
      )}>
        {renderMainContent()}
      </main>
    </div>
  );
};
