import  { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { integrationService } from '@/services/integrationService';
import { toast } from '@/hooks/use-toast';
import { Github, Slack } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

type SyncStatus = {
  notion: 'idle' | 'loading' | 'success' | 'error';
  github: 'idle' | 'loading' | 'success' | 'error';
  slack: 'idle' | 'loading' | 'success' | 'error';
};

export const IntegrationsPage = () => {
  const [syncStatus, setSyncStatus] = useState<SyncStatus>({
    notion: 'idle',
    github: 'idle',
    slack: 'idle'
  });
  
  // Notion sync mutation
  const notionSyncMutation = useMutation({
    mutationFn: integrationService.syncNotionData,
    onMutate: () => {
      setSyncStatus(prev => ({ ...prev, notion: 'loading' }));
      toast({
        title: "Notion Sync",
        description: "Starting Notion data synchronization..."
      });
    },
    onSuccess: () => {
      setSyncStatus(prev => ({ ...prev, notion: 'success' }));
      toast({
        title: "Notion Sync",
        description: "Successfully synchronized Notion data!"
      });
    },
    onError: (error: any) => {
      setSyncStatus(prev => ({ ...prev, notion: 'error' }));
      toast({
        title: "Notion Sync Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  // GitHub sync mutation
  const githubSyncMutation = useMutation({
    mutationFn: integrationService.syncGithubData,
    onMutate: () => {
      setSyncStatus(prev => ({ ...prev, github: 'loading' }));
      toast({
        title: "GitHub Sync",
        description: "Starting GitHub data synchronization..."
      });
    },
    onSuccess: () => {
      setSyncStatus(prev => ({ ...prev, github: 'success' }));
      toast({
        title: "GitHub Sync",
        description: "Successfully synchronized GitHub data!"
      });
    },
    onError: (error: any) => {
      setSyncStatus(prev => ({ ...prev, github: 'error' }));
      toast({
        title: "GitHub Sync Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  // Slack sync mutation
  const slackSyncMutation = useMutation({
    mutationFn: integrationService.syncSlackData,
    onMutate: () => {
      setSyncStatus(prev => ({ ...prev, slack: 'loading' }));
      toast({
        title: "Slack Sync",
        description: "Starting Slack data synchronization..."
      });
    },
    onSuccess: () => {
      setSyncStatus(prev => ({ ...prev, slack: 'success' }));
      toast({
        title: "Slack Sync",
        description: "Successfully synchronized Slack data!"
      });
    },
    onError: (error: any) => {
      setSyncStatus(prev => ({ ...prev, slack: 'error' }));
      toast({
        title: "Slack Sync Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-700">
        <div>
          <h1 className="text-2xl font-semibold text-white">Manage Integrations & Data Sync</h1>
          <p className="text-gray-400">Connect and synchronize your data from external platforms</p>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-auto">
        <div className="grid gap-6">
          {/* Data Syncs */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-xl text-white">Data Synchronization</CardTitle>
              <CardDescription className="text-gray-400">
                Sync your data from external platforms
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Notion */}
              <div className="flex items-center justify-between p-4 bg-gray-850 rounded-md">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-purple-900 rounded-md flex items-center justify-center">
                    <span className="text-purple-200 text-lg font-bold">N</span>
                  </div>
                  <div>
                    <h3 className="text-white font-medium">Notion</h3>
                    <p className="text-xs text-gray-400">Sync your Notion documents and databases</p>
                  </div>
                </div>
                <Button 
                  disabled={syncStatus.notion === 'loading'}
                  onClick={() => notionSyncMutation.mutate()}
                  className="min-w-24 border border-gray-600 bg-transparent hover:bg-gray-700"
                >
                  {syncStatus.notion === 'loading' ? 'Syncing...' : 'Sync Now'}
                </Button>
              </div>

              {/* GitHub */}
              <div className="flex items-center justify-between p-4 bg-gray-850 rounded-md">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gray-700 rounded-md flex items-center justify-center">
                    <Github className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-white font-medium">GitHub</h3>
                    <p className="text-xs text-gray-400">Sync your repositories and issues</p>
                  </div>
                </div>
                <Button 
                  disabled={syncStatus.github === 'loading'}
                  onClick={() => githubSyncMutation.mutate()}
                  className="min-w-24 border border-gray-600 bg-transparent hover:bg-gray-700"
                >
                  {syncStatus.github === 'loading' ? 'Syncing...' : 'Sync Now'}
                </Button>
              </div>

              {/* Slack */}
              <div className="flex items-center justify-between p-4 bg-gray-850 rounded-md">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-green-900 rounded-md flex items-center justify-center">
                    <Slack className="w-5 h-5 text-green-200" />
                  </div>
                  <div>
                    <h3 className="text-white font-medium">Slack</h3>
                    <p className="text-xs text-gray-400">Sync your channels and messages</p>
                  </div>
                </div>
                <Button 
                  disabled={syncStatus.slack === 'loading'}
                  onClick={() => slackSyncMutation.mutate()}
                  className="min-w-24 border border-gray-600 bg-transparent hover:bg-gray-700"
                >
                  {syncStatus.slack === 'loading' ? 'Syncing...' : 'Sync Now'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
