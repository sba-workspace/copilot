#notion_tools.py
import os
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field, model_validator
from langchain_core.tools import BaseTool
from notion_client import Client
import logging

logger = logging.getLogger(__name__)

def get_notion_client() -> Client:
    """Initialize and return a Notion client with better error handling."""
    notion_api_key = os.getenv("NOTION_API_KEY")
    if not notion_api_key:
        raise ValueError("NOTION_API_KEY environment variable not set. Please configure your Notion integration.")
    
    try:
        client = Client(auth=notion_api_key)
        # Test the connection
        client.users.me()
        return client
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            raise ValueError("Invalid Notion API key. Please check your NOTION_API_KEY configuration.")
        elif "404" in error_msg:
            raise ValueError("Notion API endpoint not found. Please check your integration configuration.")
        else:
            raise ValueError(f"Notion client initialization failed: {error_msg}")

def extract_database_title(database: dict) -> str:
    """Extract the title from a Notion database object"""
    try:
        # Check if title exists and is not empty
        if "title" in database and isinstance(database["title"], list) and len(database["title"]) > 0:
            # Get the first title element
            title_element = database["title"][0]
            # Try to get plain_text first (simplest)
            if "plain_text" in title_element:
                return title_element["plain_text"]
            # If no plain_text, try to get from text.content
            elif "text" in title_element and "content" in title_element["text"]:
                return title_element["text"]["content"]
        # If we can't find a proper title, try to get from other fields
        if "properties" in database:
            # Look for a title property
            for prop_name, prop_value in database["properties"].items():
                if prop_value.get("type") == "title" and prop_value.get("title"):
                    # Get the first text element's content
                    if prop_value["title"] and len(prop_value["title"]) > 0:
                        return prop_value["title"][0].get("plain_text", "Untitled")
    except Exception as e:
        logger.error(f"Error extracting database title: {str(e)}")
    
    # Default to "Untitled" if we can't find a proper title
    return "Untitled"

class NotionBaseTool(BaseTool):
    """Base class for Notion tools with enhanced error handling."""
    notion_client: Optional[Client] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.notion_client = get_notion_client()
        except Exception as e:
            logger.error(f"Notion client initialization failed: {str(e)}")
            raise
    
    def _handle_notion_error(self, e: Exception, operation: str) -> Dict[str, Any]:
        """Standardized error handling for Notion operations."""
        error_msg = str(e)
        logger.error(f"Notion {operation} failed: {error_msg}")
        
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return {
                "error": "Authentication failed: Invalid Notion API key. Please check your configuration.",
                "is_valid": False,
                "auth_error": True
            }
        elif "404" in error_msg:
            return {
                "error": "Resource not found. Please check the database or page ID.",
                "is_valid": False,
                "not_found": True
            }
        elif "400" in error_msg or "invalid" in error_msg.lower():
            return {
                "error": f"Invalid request: {error_msg}",
                "is_valid": False,
                "validation_error": True
            }
        else:
            return {
                "error": f"Operation failed: {error_msg}",
                "is_valid": False
            }
            
            
class ListDatabasesInput(BaseModel):
    query: Optional[str] = Field(None, description="Search query to filter databases")
    # limit: Optional[int] = Field(20, description="Maximum number of results to return")
    
class ListDatabasesTool(NotionBaseTool):
    name: str = "list_notion_databases"
    description: str = "List Notion databases with optional search filter"
    args_schema: Type[BaseModel] = ListDatabasesInput
    
    def _run(self, query: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Get all databases with pagination handling
            all_databases = []
            has_more = True
            next_cursor = None
            
            while has_more:
                params = {
                    "filter": {"property": "object", "value": "database"}
                }
                if query:
                    params["query"] = query
                if next_cursor:
                    params["start_cursor"] = next_cursor
                    
                response = self.notion_client.search(**params)
                all_databases.extend(response.get("results", []))
                has_more = response.get("has_more", False)
                next_cursor = response.get("next_cursor")
            
            # Format the results to include a proper name field
            formatted_databases = []
            for db in all_databases:
                title = extract_database_title(db)
                # Apply client-side filtering if query is provided
                if query and query.strip() and query.lower() not in title.lower():
                    continue
                formatted_databases.append({
                    "name": title,
                    "id": db["id"],
                    "database_id": db["id"],
                    "url": db["url"]
                })
            
            return {"databases": formatted_databases}
        except Exception as e:
            logger.error(f"Error listing databases: {str(e)}")
            return {"error": str(e)}
        
class GetDatabaseInput(BaseModel):
    database_id: Optional[str] = None
    
class GetDatabaseTool(NotionBaseTool):
    name: str = "get_notion_database"
    description: str = "Get details of a specific Notion database"
    args_schema: Type[BaseModel] = GetDatabaseInput
    
    def _run(self, database_id: Optional[str] = None,) -> Dict[str, Any]:
        db_id = database_id 
        if not db_id:
            # Return a friendly, structured error so the agent/user sees what went wrong
            err = "Missing required parameter: database_id (or id). Please provide a database id."
            logger.warning("get_notion_database called without id/database_id")
            return {"error": err, "is_valid": False, "message_for_user": err}
        try:
            database = self.notion_client.databases.retrieve(database_id=db_id)
            return {"database": database}
        except Exception as e:
            logger.error(f"Error retrieving database: {str(e)}")
            return {"error": str(e)}

class QueryDatabaseInput(BaseModel):
    database_id: str = Field(..., description="The ID of the database to query")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter conditions for the query")
    sorts: Optional[Dict[str, Any]] = Field(None, description="Sort conditions for the query")
    page_size: Optional[int] = Field(10, description="Number of entries to return (default: 10, max: 100)")
    
class QueryDatabaseTool(NotionBaseTool):
    name: str = "query_notion_database"
    description: str = "Query a Notion database with filters and sorts"
    args_schema: Type[BaseModel] = QueryDatabaseInput
    
    def _run(self, database_id: str, filter: Optional[Dict[str, Any]] = None, sorts: Optional[Dict[str, Any]] = None, page_size: Optional[int] = 10) -> Dict[str, Any]:
        try:
            query_params = {"database_id": database_id, "page_size": min(page_size or 10, 100)}
            if filter:
                query_params["filter"] = filter
            if sorts:
                query_params["sorts"] = sorts
                
            response = self.notion_client.databases.query(**query_params)
            return response  # Return the full response, not just results
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            return self._handle_notion_error(e, "database query")
        
class CreatePageInput(BaseModel):
    database_id: str = Field(..., description="The ID of the database to create the page in")
    properties: Dict[str, Any] = Field(..., description="Properties for the new page")
    content: Optional[str] = Field(None, description="Content for the new page")

class CreatePageTool(NotionBaseTool):
    name: str = "create_notion_page"
    description: str = "Create a new page in a Notion database"
    args_schema: Type[BaseModel] = CreatePageInput
    
    def _run(self, database_id: str, properties: Dict[str, Any], content: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Prepare page creation parameters
            page_params = {
                "parent": {"database_id": database_id},
                "properties": properties
            }
            
            # Add content if provided
            if content:
                page_params["children"] = [
                    {
                        "object": "block",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": content}}]
                        }
                    }
                ]
            
            page = self.notion_client.pages.create(**page_params)
            return {"page": page}
        except Exception as e:
            logger.error(f"Error creating page: {str(e)}")
            return self._handle_notion_error(e, "create query")
        
class GetPageInput(BaseModel):
    page_id: str = Field(..., description="The ID of the page to retrieve")

class GetPageTool(NotionBaseTool):
    name: str = "get_notion_page"
    description: str = "Get details of a specific Notion page"
    args_schema: Type[BaseModel] = GetPageInput
    
    def _run(self, page_id: str) -> Dict[str, Any]:
        try:
            page = self.notion_client.pages.retrieve(page_id=page_id)
            return {"page": page}
        except Exception as e:
            logger.error(f"Error retrieving page: {str(e)}")
            return self._handle_notion_error(e, "get page query")
        
class UpdatePageInput(BaseModel):
    page_id: str = Field(..., description="The ID of the page to update")
    properties: Dict[str, Any] = Field(..., description="Properties to update on the page")

class UpdatePageTool(NotionBaseTool):
    name: str = "update_notion_page"
    description: str = "Update properties of a Notion page"
    args_schema: Type[BaseModel] = UpdatePageInput
    
    def _run(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        try:
            page = self.notion_client.pages.update(
                page_id=page_id,
                properties=properties
            )
            return {"page": page}
        except Exception as e:
            logger.error(f"Error updating page: {str(e)}")
            return self._handle_notion_error(e, "update page query")
        
class SearchNotionInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")

class SearchNotionTool(NotionBaseTool):
    name: str = "search_notion"
    description: str = "Search across all Notion pages and databases"
    args_schema: Type[BaseModel] = SearchNotionInput
    
    def _run(self, query: str, limit: Optional[int] = 10) -> Dict[str, Any]:
        try:
            response = self.notion_client.search(
                query=query,
                sort={
                    "direction": "descending",
                    "timestamp": "last_edited_time"
                }
            )
            results = response.get("results", [])
            # Limit results if specified
            if limit and len(results) > limit:
                results = results[:limit]
            return {"results": results}
        except Exception as e:
            logger.error(f"Error searching Notion: {str(e)}")
            return self._handle_notion_error(e, "search query")
        
def get_notion_tools() -> list:
    """Get all Notion tools."""
    return [
        ListDatabasesTool(),
        GetDatabaseTool(),
        QueryDatabaseTool(),
        CreatePageTool(),
        GetPageTool(),
        UpdatePageTool(),
        SearchNotionTool()
    ]