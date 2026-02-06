# executor_mcp.py - MCP-enabled task executor
import asyncio
import logging
import re
from typing import Dict, Any, Optional

from app.agents.utils import AgentState, TaskStatus
from app.mcp.mcp_client import get_mcp_client, initialize_mcp_clients

logger = logging.getLogger(__name__)

def extract_parameters_from_query(user_query: str, task_description: str) -> Dict[str, Any]:
    """Extract parameters from user query and task description"""
    logger.info(f"=== PARAMETER EXTRACTION START ===")
    logger.info(f"User query: {user_query}")
    logger.info(f"Task description: {task_description}")
    
    params = {}
    
    # Extract issue numbers
    issue_match = re.search(r'issue\s*(?:number\s*)?#?(\d+)', user_query.lower())
    if issue_match:
        params["issue_number"] = int(issue_match.group(1))
        logger.info(f"Extracted issue_number: {params['issue_number']}")
    
    # Extract PR numbers
    pr_match = re.search(r'(?:pr|pull\s*request)\s*(?:number\s*)?#?(\d+)', user_query.lower())
    if pr_match:
        params["pr_number"] = int(pr_match.group(1))
        logger.info(f"Extracted pr_number: {params['pr_number']}")
    
    # Extract file paths
    file_match = re.search(r'(?:file\s*(?:called\s*)?["\']([^"\']+)[\'"]|([a-zA-Z0-9_.-]+\.[a-zA-Z]+))', user_query)
    if file_match:
        params["file_path"] = file_match.group(1) or file_match.group(2)
        logger.info(f"Extracted file_path: {params['file_path']}")
    
    # Extract search queries
    if "search" in task_description.lower():
        search_match = re.search(r'["\']([^"\']+)[\'"]', user_query)
        if search_match:
            params["query"] = search_match.group(1)
            logger.info(f"Extracted search query: {params['query']}")
    
    # Extract file content (handles multi-line content)
    content_match = re.search(r'with\s+content\s+(["\'])([\s\S]*?)\1', user_query)
    if content_match:
        params["content"] = content_match.group(2)
        logger.info(f"Extracted content: {params['content'][:100]}...")
    
    # Extract commit message
    message_match = re.search(r'(?:with\s+message|commit\s+message)\s+(["\'])([\s\S]*?)\1', user_query)
    if message_match:
        params["commit_message"] = message_match.group(2)
        logger.info(f"Extracted commit message: {params['commit_message']}")
    
    # Extract comments
    if "comment" in task_description.lower():
        comment_match = re.search(r'saying\s*["\']([^"\']+)[\'"]', user_query)
        if comment_match:
            params["comment"] = comment_match.group(1)
            logger.info(f"Extracted comment: {params['comment']}")
    
    # Database name extraction for Notion operations
    if "database" in task_description.lower() or "database" in user_query.lower():
        logger.info("Database operation detected, extracting database information...")
        
        # Extract database name using multiple patterns
        patterns = [
            r"(?:show\s+entries\s+in\s+the\s+|entries\s+in\s+the\s+|list\s+all\s+items\s+in\s+the\s+|get\s+database\s+contents\s+for\s+)([A-Za-z0-9\s_-]+?)(?:\s+database|$)",
            r"(?:get\s+details\s+of\s+the\s+database|database\s+details\s+for)\s+([A-Za-z0-9\s_-]+?)(?:\s*$|[.!?])",
            r"query\s+the\s+['\"]([^'\"]+)['\"]\s+database",
            r"([A-Za-z][A-Za-z0-9\s_-]*?)\s+database"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                params["database_name"] = match.group(1).strip()
                logger.info(f"Extracted database_name: {params['database_name']}")
                break
        
        # Extract database ID if provided
        uuid_match = re.search(r'(?:with\s+ID|ID\s*[:=])\s*[\'"]?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})[\'"]?', user_query, re.IGNORECASE)
        if uuid_match:
            params["database_id"] = uuid_match.group(1)
            logger.info(f"Extracted database_id: {params['database_id']}")
    
    logger.info(f"Final extracted parameters: {params}")
    logger.info(f"=== PARAMETER EXTRACTION END ===")
    return params

async def async_mcp_task_executor(state: AgentState) -> Dict[str, Any]:
    """Async MCP-enabled task executor"""
    logger.info("=== MCP TASK EXECUTOR START ===")
    logger.info(f"Current state keys: {list(state.keys())}")
    
    task_queue = state["task_queue"]
    current_index = state["current_task_index"]
    task_results = state["task_results"].copy()
    shared_context = state["shared_context"].copy()
    execution_status = state["execution_status"].copy()
    errors = state["errors"].copy()
    
    logger.info(f"Current task index: {current_index}, Total tasks: {len(task_queue)}")
    
    if current_index >= len(task_queue):
        logger.info("All tasks completed")
        return {
            **state,
            "is_complete": True,
            "current_task_index": current_index
        }
    
    current_task = task_queue[current_index]
    task_id = current_task["task_id"]
    tool_name = current_task["tool_name"]
    service = current_task.get("mcp_service", current_task.get("service", "github"))
    description = current_task["description"]
    
    logger.info(f"=== EXECUTING MCP TASK {current_index + 1}/{len(task_queue)} ===")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"MCP Service: {service}")
    logger.info(f"Tool name: {tool_name}")
    logger.info(f"Description: {description}")
    
    # Initialize MCP clients if not already done
    try:
        mcp_client = await get_mcp_client()
        
        # Ensure MCP clients are initialized
        if not mcp_client.github_session or not mcp_client.notion_session:
            logger.info("Initializing MCP clients...")
            init_results = await initialize_mcp_clients()
            logger.info(f"MCP initialization results: {init_results}")
        
        # Extract parameters from the task description and user query
        user_query = state.get("user_query", "")
        parameters = extract_parameters_from_query(user_query, description)
        
        # Map common parameter variations to tool-specific names
        param_mapping = get_parameter_mapping(tool_name)
        mapped_params = {}
        for key, value in parameters.items():
            mapped_key = param_mapping.get(key, key)
            mapped_params[mapped_key] = value
        
        logger.info(f"Calling MCP {service}.{tool_name} with parameters: {mapped_params}")
        
        # Call the appropriate MCP service
        if service == "github":
            result = await mcp_client.call_github_tool(tool_name, mapped_params)
        elif service == "notion":
            result = await mcp_client.call_notion_tool(tool_name, mapped_params)
        else:
            result = {"error": f"Unknown MCP service: {service}"}
        
        logger.info(f"MCP call result: {result}")
        
        # Store results
        task_results[task_id] = result
        
        # Update execution status
        if "error" in result:
            execution_status[task_id] = TaskStatus.FAILED.value
            errors.append(f"Task {task_id} failed: {result['error']}")
            logger.error(f"Task {task_id} failed: {result['error']}")
        else:
            execution_status[task_id] = TaskStatus.COMPLETED.value
            logger.info(f"Task {task_id} completed successfully")
        
        # Update shared context if needed
        shared_context = update_shared_context(shared_context, service, tool_name, result)
        
    except Exception as e:
        logger.error(f"MCP task execution failed: {str(e)}")
        task_results[task_id] = {"error": str(e)}
        execution_status[task_id] = TaskStatus.FAILED.value
        errors.append(f"Task {task_id} failed: {str(e)}")
    
    return {
        **state,
        "task_results": task_results,
        "shared_context": shared_context,
        "execution_status": execution_status,
        "errors": errors,
        "current_task_index": current_index + 1
    }

def mcp_task_executor(state: AgentState) -> Dict[str, Any]:
    """Synchronous wrapper for MCP task executor (required for LangGraph)"""
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to use a different approach
            # For now, we'll create a new event loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_mcp_task_executor(state))
                return future.result()
        else:
            # No running loop, we can use asyncio.run
            return asyncio.run(async_mcp_task_executor(state))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Handle the case where we're in an async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_mcp_task_executor(state))
                return future.result()
        else:
            raise

def get_parameter_mapping(tool_name: str) -> Dict[str, str]:
    """Get parameter name mapping for specific tools"""
    mappings = {
        "create_pull_request": {
            "description": "body",
            "from_branch": "head",
            "to_branch": "base",
            "source_branch": "head",
            "target_branch": "base"
        },
        "comment_on_issue": {
            "text": "comment"
        },
        "create_file": {
            "path": "file_path",
            "message": "commit_message"
        },
        "create_notion_page": {
            "database_name": "database_id"  # Will be resolved later
        },
        "query_notion_database": {
            "database_name": "database_id"  # Will be resolved later
        }
    }
    return mappings.get(tool_name, {})

def update_shared_context(shared_context: Dict[str, Any], service: str, tool_name: str, result: Any) -> Dict[str, Any]:
    """Update shared context with relevant information from MCP tool execution"""
    logger.info(f"Updating shared context for MCP {service}.{tool_name}")
    
    if service == "github":
        if tool_name == "get_issue" and isinstance(result, dict) and "number" in result:
            shared_context["last_issue"] = result
        elif tool_name == "get_pull_request" and isinstance(result, dict) and "number" in result:
            shared_context["last_pull_request"] = result
        elif tool_name == "create_file" and isinstance(result, dict) and result.get("success"):
            shared_context["last_created_file"] = result
    
    elif service == "notion":
        if tool_name == "list_notion_databases" and isinstance(result, dict) and "databases" in result:
            # Store database mappings for future use
            db_map = {}
            for db in result["databases"]:
                db_name = db.get("name", db.get("title", "Untitled"))
                db_map[db_name.lower()] = db["id"]
            shared_context["database_id_map"] = db_map
        elif tool_name == "get_notion_database" and isinstance(result, dict) and "database" in result:
            shared_context["last_database"] = result["database"]
        elif tool_name == "query_notion_database" and isinstance(result, dict) and "results" in result:
            shared_context["last_query_results"] = result["results"]
    
    return shared_context

# Backward compatibility
task_executor = mcp_task_executor
