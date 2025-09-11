import { FC } from 'react';
import { MessageSquare, CheckSquare, Settings, Menu, Bot, LogOut, UserCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ViewType } from './Layout';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

interface SidebarProps {
  currentView: ViewType;
  onViewChange: (view: ViewType) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: FC<SidebarProps> = ({
  currentView,
  onViewChange,
  collapsed,
  onToggleCollapse
}) => {
  const { signOut, user } = useAuth();
  const navigate = useNavigate();
  
  const navigationItems = [
    { id: 'chat' as ViewType, icon: MessageSquare, label: 'Chat' },
    { id: 'tasks' as ViewType, icon: CheckSquare, label: 'Tasks' },
    { id: 'integrations' as ViewType, icon: Settings, label: 'Integrations' },
  ];

  return (
    <div className={cn(
      "fixed left-0 top-0 h-full bg-gray-800 border-r border-gray-700 transition-all duration-300 z-50",
      collapsed ? "w-16" : "w-64"
    )}>
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className={cn(
            "flex items-center space-x-3 transition-opacity duration-300",
            collapsed ? "opacity-0" : "opacity-100"
          )}>
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold">AI Copilot</span>
          </div>
          <button
            onClick={onToggleCollapse}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            aria-label="Toggle sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.id}>
                  <button
                    onClick={() => onViewChange(item.id)}
                    className={cn(
                      "w-full flex items-center space-x-3 px-3 py-3 rounded-lg transition-all duration-200",
                      currentView === item.id
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:bg-gray-700 hover:text-white"
                    )}
                    title={collapsed ? item.label : undefined}
                  >
                    <Icon className="w-5 h-5 flex-shrink-0" />
                    <span className={cn(
                      "transition-opacity duration-300",
                      collapsed ? "opacity-0 w-0" : "opacity-100"
                    )}>
                      {item.label}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <div className={cn(
              "flex items-center space-x-3 text-sm text-gray-400 transition-opacity duration-300",
              collapsed ? "opacity-0" : "opacity-100"
            )}>
              <Avatar className="w-8 h-8">
                <AvatarImage src={user?.user_metadata?.avatar_url} alt="Profile picture" />
                <AvatarFallback className="bg-gray-600 text-white text-xs font-semibold">
                  {user?.email?.charAt(0).toUpperCase() || 'U'}
                </AvatarFallback>
              </Avatar>
              <span className="truncate max-w-32">{user?.email || 'User'}</span>
            </div>
            <button
              onClick={() => signOut()}
              className="p-2 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-white transition-colors"
              title="Sign out"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
          
          {/* Profile Link */}
          <button
            onClick={() => navigate('/profile')}
            className="w-full flex items-center justify-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white"
          >
            <UserCircle className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span>View Profile</span>}
          </button>
        </div>
      </div>
    </div>
  );
};
