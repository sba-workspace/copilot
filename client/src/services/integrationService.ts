import { supabase } from '@/integrations/supabase/client';

const API_BASE_URL = '/api';

export const integrationService = {
  async syncNotionData() {
    const { data: { session } } = await supabase.auth.getSession();
    
    const response = await fetch(`${API_BASE_URL}/tools/notion-sync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to sync Notion data');
    }

    return response.json();
  },

  async syncGithubData() {
    const { data: { session } } = await supabase.auth.getSession();
    
    const response = await fetch(`${API_BASE_URL}/tools/github-sync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to sync GitHub data');
    }

    return response.json();
  },

  async syncSlackData() {
    const { data: { session } } = await supabase.auth.getSession();
    
    const response = await fetch(`${API_BASE_URL}/tools/slack-sync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to sync Slack data');
    }

    return response.json();
  },

  async semanticSearch(query: string, sources: string[], limit: number) {
    const { data: { session } } = await supabase.auth.getSession();

    const response = await fetch(`${API_BASE_URL}/tools/semantic-search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
      },
      body: JSON.stringify({ query, sources, limit })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to perform semantic search');
    }

    return response.json();
  }
};
