import { useState } from 'react';
import { Search } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { integrationService } from '@/services/integrationService';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';

interface SearchResult {
  content: string;
  source: string;
  relevance: number;
  metadata?: any;
}

export const SemanticSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedSources, setSelectedSources] = useState({
    notion: true,
    github: true,
    slack: true
  });
  const [searchLimit, setSearchLimit] = useState(5);

  const searchMutation = useMutation({
    mutationFn: async () => {
      const sources = Object.entries(selectedSources)
        .filter(([_, isSelected]) => isSelected)
        .map(([source]) => source);

      return integrationService.semanticSearch(searchQuery, sources, searchLimit);
    },
    onSuccess: (data) => {
      setSearchResults(data.results || []);
      if (data.results?.length === 0) {
        toast({
          title: "No Results",
          description: "No matching results found for your search query."
        });
      }
    },
    onError: (error: any) => {
      toast({
        title: "Search Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const handleSearch = () => {
    if (!searchQuery.trim()) {
      toast({
        title: "Invalid Search",
        description: "Please enter a search query",
        variant: "destructive"
      });
      return;
    }
    searchMutation.mutate();
  };

  return (
    <div className="space-y-6">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="text-xl text-white">Semantic Search</CardTitle>
          <CardDescription className="text-gray-400">
            Search across your connected platforms
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search Input */}
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
              disabled={!searchQuery.trim() || searchMutation.isPending}
            >
              {searchMutation.isPending ? "Searching..." : "Search"}
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
            <div className="flex flex-col space-y-1 mt-2">
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
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-lg text-white">Search Results</CardTitle>
            <CardDescription className="text-gray-400">
              {searchResults.length} results found for "{searchQuery}"
            </CardDescription>
          </CardHeader>
          <CardContent>
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
                  {result.metadata && (
                    <div className="mt-1 text-xs text-gray-400">
                      {Object.entries(result.metadata).map(([key, value]) => (
                        <span key={key} className="mr-3">
                          {key}: {String(value)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}; 