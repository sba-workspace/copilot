import  { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { integrationService } from '@/services/integrationService';
import { toast } from '@/hooks/use-toast';
import { Github, Search, Slack } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

type SyncStatus = {
  notion: 'idle' | 'loading' | 'success' | 'error';
  github: 'idle' | 'loading' | 'success' | 'error';
  slack: 'idle' | 'loading' | 'success' | 'error';
};

type SearchResult = {
  content: string;
  source: string;
  relevance: number;
  metadata?: any;
};

export const IntegrationHub = () => {
  const [syncStatus, setSyncStatus] = useState<SyncStatus>({
    notion: 'idle',
    github: 'idle',
    slack: 'idle'
  });
  
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedSources, setSelectedSources] = useState({
    notion: true,
    github: true,
    slack: true
  });
  const [searchLimit, setSearchLimit] = useState(5);

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

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setIsSearching(true);
      
      const sources = [];
      if (selectedSources.notion) sources.push('notion');
      if (selectedSources.github) sources.push('github');
      if (selectedSources.slack) sources.push('slack');
      
      const results = await integrationService.semanticSearch(
        searchQuery,
        sources,
        searchLimit
      );
      
      setSearchResults(results || []);
    } catch (error: any) {
      toast({
        title: "Search Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-700">
        <div>
          <h1 className="text-2xl font-semibold text-white">Integration Hub</h1>
          <p className="text-gray-400">Connect and search across your data sources</p>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Data Syncs */}
          <div className="space-y-6">
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
                    className="min-w-24"
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
                    className="min-w-24"
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
                    className="min-w-24"
                  >
                    {syncStatus.slack === 'loading' ? 'Syncing...' : 'Sync Now'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Semantic Search */}
          <div>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-xl text-white">Semantic Search</CardTitle>
                <CardDescription className="text-gray-400">
                  Search across your connected platforms
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <div className="relative flex-1">
                    <Search className="absolute top-2.5 left-3 h-4 w-4 text-gray-500" />
                    <Input
                      type="text"
                      placeholder="Search across your data sources..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 bg-gray-700 border-gray-600 text-white placeholder-gray-400"
                    />
                  </div>
                  <Button 
                    onClick={handleSearch}
                    disabled={!searchQuery.trim() || isSearching}
                  >
                    {isSearching ? "Searching..." : "Search"}
                  </Button>
                </div>

                {/* Search Options */}
                <div className="flex flex-col space-y-2 p-4 bg-gray-850 rounded-md">
                  <h3 className="text-white font-medium mb-2">Search Options</h3>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="notion"
                      checked={selectedSources.notion} 
                      onCheckedChange={(checked) => 
                        setSelectedSources(prev => ({ ...prev, notion: !!checked }))
                      }
                    />
                    <label htmlFor="notion" className="text-sm text-gray-300">Notion</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="github"
                      checked={selectedSources.github} 
                      onCheckedChange={(checked) => 
                        setSelectedSources(prev => ({ ...prev, github: !!checked }))
                      }
                    />
                    <label htmlFor="github" className="text-sm text-gray-300">GitHub</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="slack"
                      checked={selectedSources.slack} 
                      onCheckedChange={(checked) => 
                        setSelectedSources(prev => ({ ...prev, slack: !!checked }))
                      }
                    />
                    <label htmlFor="slack" className="text-sm text-gray-300">Slack</label>
                  </div>
                  <Separator className="my-2" />
                  <div className="flex flex-col space-y-1">
                    <label htmlFor="limit" className="text-sm text-gray-300">Results Limit</label>
                    <Input
                      id="limit"
                      type="number"
                      min={1}
                      max={20}
                      value={searchLimit}
                      onChange={(e) => setSearchLimit(Number(e.target.value))}
                      className="bg-gray-700 border-gray-600 text-white w-24"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <Card className="bg-gray-800 border-gray-700 mt-6">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg text-white">Search Results</CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="divide-y divide-gray-700">
                    {searchResults.map((result, index) => (
                      <div key={index} className="py-3">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-xs px-2 py-1 bg-blue-900 text-blue-100 rounded-sm">
                            {result.source}
                          </span>
                          <span className="text-xs text-gray-400">
                            Relevance: {result.relevance.toFixed(2)}
                          </span>
                        </div>
                        <p className="text-white text-sm">{result.content}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
                <CardFooter>
                  <div className="text-xs text-gray-400">
                    {searchResults.length} results found for "{searchQuery}"
                  </div>
                </CardFooter>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
