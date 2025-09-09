"""
Notion MCP Server
Provides Notion functionality through Model Context Protocol
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
load_dotenv()

# MCP imports will be added once the package is installed
# For now, we'll use placeholders
try:
    from mcp import Server
    from mcp import InitializationOptions
    from mcp import (
        CallToolRequest,
        CallToolResult,
        ListToolsRequest,
        ListToolsResult,
        Tool,
        TextContent
    )
except ImportError:
    # Placeholder classes for development
    class Server:
        def __init__(self, name): 
            self.name = name
        def list_tools(self): 
            def decorator(func):
                return func
            return decorator
        def call_tool(self): 
            def decorator(func):
                return func
            return decorator
    
    class Tool:
        def __init__(self, **kwargs): 
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ListToolsResult:
        def __init__(self, tools): self.tools = tools
    
    class CallToolResult:
        def __init__(self, content): self.content = content
    
    class TextContent:
        def __init__(self, type, text): self.type = type; self.text = text
from pydantic import BaseModel
from notion_client import Client
from notion_client.errors import APIResponseError

logger = logging.getLogger(__name__)

class NotionMCPServer:
    """Notion MCP Server implementation"""
    
    def __init__(self):
        self.app = Server("notion-mcp")
        self.notion_client: Optional[Client] = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP handlers"""
        
        @self.app.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available Notion tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="list_notion_databases",
                        description="List Notion databases with optional search filter",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query to filter databases"}
                            },
                            "required": []
                        }
                    ),
                    Tool(
                        name="get_notion_database",
                        description="Get details of a specific Notion database",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "database_id": {"type": "string", "description": "Database ID"}
                            },
                            "required": ["database_id"]
                        }
                    ),
                    Tool(
                        name="query_notion_database",
                        description="Query a Notion database with filters and sorts",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "database_id": {"type": "string", "description": "Database ID"},
                                "filter": {"type": "object", "description": "Filter conditions"},
                                "sorts": {"type": "object", "description": "Sort conditions"},
                                "page_size": {"type": "integer", "description": "Number of entries to return", "default": 10}
                            },
                            "required": ["database_id"]
                        }
                    ),
                    Tool(
                        name="create_notion_page",
                        description="Create a new page in a Notion database",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "database_id": {"type": "string", "description": "Database ID"},
                                "properties": {"type": "object", "description": "Page properties"},
                                "content": {"type": "string", "description": "Page content"}
                            },
                            "required": ["database_id", "properties"]
                        }
                    ),
                    Tool(
                        name="get_notion_page",
                        description="Get details of a specific Notion page",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "page_id": {"type": "string", "description": "Page ID"}
                            },
                            "required": ["page_id"]
                        }
                    ),
                    Tool(
                        name="update_notion_page",
                        description="Update properties of a Notion page",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "page_id": {"type": "string", "description": "Page ID"},
                                "properties": {"type": "object", "description": "Properties to update"}
                            },
                            "required": ["page_id", "properties"]
                        }
                    ),
                    Tool(
                        name="search_notion",
                        description="Search across all Notion pages and databases",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "limit": {"type": "integer", "description": "Maximum number of results", "default": 10}
                            },
                            "required": ["query"]
                        }
                    )
                ]
            )
        
        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Execute a Notion tool"""
            try:
                if not self._ensure_notion_client():
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Notion client not initialized. Please check your NOTION_API_KEY configuration."
                        )]
                    )
                
                result = await self._execute_tool(name, arguments)
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(result, default=str, indent=2)
                    )]
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error executing tool {name}: {str(e)}"
                    )]
                )

    def _ensure_notion_client(self) -> bool:
        """Ensure Notion client is initialized"""
        if self.notion_client is None:
            notion_api_key = os.getenv("NOTION_API_KEY")
            if not notion_api_key:
                logger.error("NOTION_API_KEY not found in environment")
                return False
            
            try:
                self.notion_client = Client(auth=notion_api_key)
                # Test the connection
                self.notion_client.users.me()
            except Exception as e:
                logger.error(f"Failed to initialize Notion client: {str(e)}")
                return False
        
        return True
    
    def _extract_database_title(self, database: dict) -> str:
        """Extract the title from a Notion database object"""
        try:
            if "title" in database and isinstance(database["title"], list) and len(database["title"]) > 0:
                title_element = database["title"][0]
                if "plain_text" in title_element:
                    return title_element["plain_text"]
                elif "text" in title_element and "content" in title_element["text"]:
                    return title_element["text"]["content"]
            
            if "properties" in database:
                for prop_name, prop_value in database["properties"].items():
                    if prop_value.get("type") == "title" and prop_value.get("title"):
                        if prop_value["title"] and len(prop_value["title"]) > 0:
                            title_element = prop_value["title"][0]
                            if "plain_text" in title_element:
                                return title_element["plain_text"]
        except Exception as e:
            logger.error(f"Error extracting database title: {str(e)}")
        
        return "Untitled"
    
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific Notion tools"""
        
        if name == "list_notion_databases":
            return await self._list_notion_databases(arguments.get("query"))
        elif name == "get_notion_database":
            return await self._get_notion_database(arguments["database_id"])
        elif name == "query_notion_database":
            return await self._query_notion_database(
                arguments["database_id"],
                arguments.get("filter"),
                arguments.get("sorts"),
                arguments.get("page_size", 10)
            )
        elif name == "create_notion_page":
            return await self._create_notion_page(
                arguments["database_id"],
                arguments["properties"],
                arguments.get("content")
            )
        elif name == "get_notion_page":
            return await self._get_notion_page(arguments["page_id"])
        elif name == "update_notion_page":
            return await self._update_notion_page(
                arguments["page_id"],
                arguments["properties"]
            )
        elif name == "search_notion":
            return await self._search_notion(
                arguments["query"],
                arguments.get("limit", 10)
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _list_notion_databases(self, query: Optional[str] = None) -> Dict[str, Any]:
        """List Notion databases"""
        try:
            all_databases = []
            has_more = True
            next_cursor = None
            
            while has_more:
                params = {
                    "filter": {"property": "object", "value": "database"}
                }
                if next_cursor:
                    params["start_cursor"] = next_cursor
                
                response = self.notion_client.search(**params)
                all_databases.extend(response.get("results", []))
                has_more = response.get("has_more", False)
                next_cursor = response.get("next_cursor")
            
            # Filter by query if provided
            if query:
                filtered_databases = []
                query_lower = query.lower()
                for db in all_databases:
                    title = self._extract_database_title(db).lower()
                    if query_lower in title:
                        filtered_databases.append(db)
                all_databases = filtered_databases
            
            # Format the results
            formatted_databases = []
            for db in all_databases:
                title = self._extract_database_title(db)
                formatted_databases.append({
                    "id": db["id"],
                    "name": title,
                    "title": title,
                    "url": db.get("url", ""),
                    "created_time": db.get("created_time", ""),
                    "last_edited_time": db.get("last_edited_time", ""),
                    "properties": db.get("properties", {})
                })
            
            return {"databases": formatted_databases, "count": len(formatted_databases)}
        except Exception as e:
            logger.error(f"Error listing databases: {str(e)}")
            return {"error": str(e)}
    
    async def _get_notion_database(self, database_id: str) -> Dict[str, Any]:
        """Get database details"""
        try:
            database = self.notion_client.databases.retrieve(database_id=database_id)
            return {"database": database}
        except Exception as e:
            logger.error(f"Error retrieving database: {str(e)}")
            return {"error": str(e)}
    
    async def _query_notion_database(self, database_id: str, filter_obj: Optional[Dict[str, Any]] = None, 
                                   sorts: Optional[Dict[str, Any]] = None, page_size: int = 10) -> Dict[str, Any]:
        """Query database with filters"""
        try:
            query_params = {
                "database_id": database_id,
                "page_size": min(page_size, 100)
            }
            if filter_obj:
                query_params["filter"] = filter_obj
            if sorts:
                query_params["sorts"] = sorts
            
            response = self.notion_client.databases.query(**query_params)
            return response
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            return {"error": str(e)}
    
    async def _create_notion_page(self, database_id: str, properties: Dict[str, Any], 
                                content: Optional[str] = None) -> Dict[str, Any]:
        """Create a new page"""
        try:
            page_params = {
                "parent": {"database_id": database_id},
                "properties": properties
            }
            
            if content:
                page_params["children"] = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": content}
                                }
                            ]
                        }
                    }
                ]
            
            page = self.notion_client.pages.create(**page_params)
            return {"page": page, "success": True}
        except Exception as e:
            logger.error(f"Error creating page: {str(e)}")
            return {"error": str(e)}
    
    async def _get_notion_page(self, page_id: str) -> Dict[str, Any]:
        """Get page details"""
        try:
            page = self.notion_client.pages.retrieve(page_id=page_id)
            return {"page": page}
        except Exception as e:
            logger.error(f"Error retrieving page: {str(e)}")
            return {"error": str(e)}
    
    async def _update_notion_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update page properties"""
        try:
            page = self.notion_client.pages.update(
                page_id=page_id,
                properties=properties
            )
            return {"page": page, "success": True}
        except Exception as e:
            logger.error(f"Error updating page: {str(e)}")
            return {"error": str(e)}
    
    async def _search_notion(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search Notion workspace"""
        try:
            response = self.notion_client.search(
                query=query,
                sort={
                    "direction": "descending",
                    "timestamp": "last_edited_time"
                }
            )
            results = response.get("results", [])
            
            if limit and len(results) > limit:
                results = results[:limit]
            
            return {"results": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Error searching Notion: {str(e)}")
            return {"error": str(e)}

    def run(self, host: str = "localhost", port: int = 3002):
        """Run the MCP server"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

# Entry point for running as standalone server
if __name__ == "__main__":
    server = NotionMCPServer()
    server.run()
