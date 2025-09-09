"""
MCP Client for communicating with GitHub and Notion MCP servers
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

logger = logging.getLogger(__name__)

class MCPClient:
    """Client for communicating with MCP servers"""
    
    def __init__(self):
        self.github_session: Optional[Dict[str, Any]] = None
        self.notion_session: Optional[Dict[str, Any]] = None
        self.github_tools: Dict[str, Any] = {}
        self.notion_tools: Dict[str, Any] = {}
    
    async def initialize_github(self, server_url: str = "http://localhost:3001") -> bool:
        """Initialize GitHub MCP connection"""
        try:
            return await self._connect_http_server("github", server_url)
        except Exception as e:
            logger.error(f"Failed to initialize GitHub MCP client: {str(e)}")
            return False
    
    async def initialize_notion(self, server_url: str = "http://localhost:3002") -> bool:
        """Initialize Notion MCP connection"""
        try:
            return await self._connect_http_server("notion", server_url)
        except Exception as e:
            logger.error(f"Failed to initialize Notion MCP client: {str(e)}")
            return False
    
    async def _connect_http_server(self, service: str, base_url: str) -> bool:
        """Connect to HTTP-based MCP server"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # For now, we'll assume the servers are available if we can create the session
                # In a real implementation, we'd test the connection
                if service == "github":
                    self.github_session = {"type": "http", "base_url": base_url}
                    # Get available GitHub tools
                    self.github_tools = {
                        "get_issues": "Get open issues from the repository",
                        "get_issue": "Get a specific issue by number", 
                        "comment_on_issue": "Add a comment to an issue",
                        "search_issues_and_prs": "Search issues and pull requests",
                        "list_open_pull_requests": "List open pull requests",
                        "get_pull_request": "Get a specific pull request by number",
                        "list_pull_request_files": "List files changed in a pull request", 
                        "create_pull_request": "Create a new pull request",
                        "read_file": "Read a file from the repository",
                        "create_file": "Create a new file in the repository",
                        "update_file": "Update an existing file in the repository",
                        "delete_file": "Delete a file from the repository",
                        "overview_files_main_branch": "Get overview of files in main branch",
                        "overview_files_current_branch": "Get overview of files in current branch"
                    }
                else:
                    self.notion_session = {"type": "http", "base_url": base_url}
                    # Get available Notion tools
                    self.notion_tools = {
                        "list_notion_databases": "List Notion databases with optional search filter",
                        "get_notion_database": "Get details of a specific Notion database",
                        "query_notion_database": "Query a Notion database with filters and sorts",
                        "create_notion_page": "Create a new page in a Notion database",
                        "get_notion_page": "Get details of a specific Notion page",
                        "update_notion_page": "Update properties of a Notion page",
                        "search_notion": "Search across all Notion pages and databases"
                    }
                return True
        except Exception as e:
            logger.error(f"Failed to connect to {service} HTTP server: {str(e)}")
        return False
    
    async def call_github_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a GitHub tool via MCP - For now, import and call directly"""
        if not self.github_session:
            return {"error": "GitHub MCP session not initialized"}
        
        try:
            # For now, we'll call the GitHub MCP server directly
            from app.mcp.github_mcp_server import GitHubMCPServer
            server = GitHubMCPServer()
            if server._ensure_github_client():
                return await server._execute_tool(tool_name, arguments)
            else:
                return {"error": "Failed to initialize GitHub client"}
        except Exception as e:
            logger.error(f"GitHub tool call failed: {str(e)}")
            return {"error": str(e)}
    
    async def call_notion_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a Notion tool via MCP - For now, import and call directly"""
        if not self.notion_session:
            return {"error": "Notion MCP session not initialized"}
        
        try:
            # For now, we'll call the Notion MCP server directly
            from app.mcp.notion_mcp_server import NotionMCPServer
            server = NotionMCPServer()
            if server._ensure_notion_client():
                return await server._execute_tool(tool_name, arguments)
            else:
                return {"error": "Failed to initialize Notion client"}
        except Exception as e:
            logger.error(f"Notion tool call failed: {str(e)}")
            return {"error": str(e)}
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get available tools from both services"""
        return {
            "github": list(self.github_tools.keys()),
            "notion": list(self.notion_tools.keys())
        }
    
    async def close(self):
        """Close all connections"""
        if self.github_session and hasattr(self.github_session, 'close'):
            await self.github_session.close()
        if self.notion_session and hasattr(self.notion_session, 'close'):
            await self.notion_session.close()

# Global MCP client instance
mcp_client = MCPClient()

async def get_mcp_client() -> MCPClient:
    """Get initialized MCP client"""
    return mcp_client

async def initialize_mcp_clients() -> Dict[str, bool]:
    """Initialize both MCP clients"""
    results = {}
    
    # Try to initialize GitHub MCP
    try:
        results["github"] = await mcp_client.initialize_github()
    except Exception as e:
        logger.error(f"Failed to initialize GitHub MCP: {str(e)}")
        results["github"] = False
    
    # Try to initialize Notion MCP
    try:
        results["notion"] = await mcp_client.initialize_notion()
    except Exception as e:
        logger.error(f"Failed to initialize Notion MCP: {str(e)}")
        results["notion"] = False
    
    return results
