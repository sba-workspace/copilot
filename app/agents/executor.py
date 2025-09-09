# Enhanced executor.py with improved database resolution
import os
import logging
import re
import json
from typing import Dict, Any, Optional
from pydantic_core import PydanticUndefined
from langchain_core.tools import BaseTool

from app.agents.utils import AgentState, TaskStatus
from app.tools.github_tools import get_wrapper

logger = logging.getLogger(__name__)

PARAM_NAME_MAPPING = {
    # For Create Pull Request tool
    "create_pull_request": {
        "description": "body",
        "from_branch": "head",
        "to_branch": "base",
        "source_branch": "head",
        "target_branch": "base"
    },
    # For Comment tool
    "comment_on_issue": {
        "text": "comment"
    },
    # CRITICAL: For Create File tool - map to actual parameter names
    "create_file": {
        "path": "file_path",
        "message": "commit_message"
    },
    # For Notion tools
    "create_notion_page": {
        "database_id": "database_id",
        "properties": "properties",
        "content": "content"
    },
    "query_notion_database": {
        "database_id": "database_id",
        "filter": "filter",
        "sorts": "sorts"
    },
    "update_notion_page": {
        "page_id": "page_id",
        "properties": "properties"
    },
    "get_notion_page": {
        "page_id": "page_id"
    },
    "get_notion_database": {
        "database_id": "database_id"
    }
}

def extract_database_name_from_query(user_query: str, task_description: str) -> Optional[str]:
    """Enhanced database name extraction with multiple patterns"""
    logger.info("=== DATABASE NAME EXTRACTION ===")
    logger.info(f"User query: {user_query}")
    logger.info(f"Task description: {task_description}")
    
    # Pattern 1: "Show entries in the Products database" or "List all items in the Products database" or "Get database contents for DP Interview Prep"
    pattern1 = re.search(r"(?:show\s+entries\s+in\s+the\s+|entries\s+in\s+the\s+|list\s+all\s+items\s+in\s+the\s+|get\s+database\s+contents\s+for\s+)([A-Za-z0-9\s_-]+?)(?:\s+database|$)", user_query, re.IGNORECASE)
    if pattern1:
        db_name = pattern1.group(1).strip()
        logger.info(f"Found database name using pattern 1: '{db_name}'")
        return db_name
    
    # Pattern 2: "Get details of the database DP Interview Prep"
    pattern2 = re.search(r"(?:get\s+details\s+of\s+the\s+database|database\s+details\s+for)\s+([A-Za-z0-9\s_-]+?)(?:\s*$|[.!?])", user_query, re.IGNORECASE)
    if pattern2:
        db_name = pattern2.group(1).strip()
        logger.info(f"Found database name using pattern 2: '{db_name}'")
        return db_name
    
    # Pattern 3: "Query the 'DatabaseName' database"
    pattern3 = re.search(r"query\s+the\s+['\"]([^'\"]+)['\"]\s+database", user_query, re.IGNORECASE)
    if pattern3:
        db_name = pattern3.group(1).strip()
        logger.info(f"Found database name using pattern 3: '{db_name}'")
        return db_name
    
    # Pattern 4: "DatabaseName database"
    pattern4 = re.search(r"([A-Za-z][A-Za-z0-9\s_-]*?)\s+database", user_query, re.IGNORECASE)
    if pattern4:
        db_name = pattern4.group(1).strip()
        logger.info(f"Found database name using pattern 4: '{db_name}'")
        return db_name
    
    # Pattern 5: From task description - "Find database named Products"
    pattern5 = re.search(r"(?:find\s+database\s+named|database\s+named)\s+([A-Za-z0-9\s_-]+)", task_description, re.IGNORECASE)
    if pattern5:
        db_name = pattern5.group(1).strip()
        logger.info(f"Found database name using pattern 5: '{db_name}'")
        return db_name
    
    # Pattern 6: From task description
    pattern6 = re.search(r"(?:database|db)\s+['\"]?([^'\"\\s]+)['\"]?", task_description, re.IGNORECASE)
    if pattern6:
        db_name = pattern6.group(1).strip()
        logger.info(f"Found database name using pattern 6: '{db_name}'")
        return db_name
    
    logger.warning("Could not extract database name from query or task description")
    return None

async def resolve_database_id(state: AgentState, tool_map: Dict[str, BaseTool], target_db_name: str) -> Optional[str]:
    """
    Resolve database name to ID by first listing all databases if needed.
    Returns the database ID if found, None otherwise.
    """
    logger.info(f"=== RESOLVING DATABASE ID FOR: '{target_db_name}' ===")
    
    shared_context = state.get("shared_context", {})
    
    # Check if we already have database mappings
    if "database_id_map" in shared_context and shared_context["database_id_map"]:
        logger.info("Database mappings already available in shared context")
        for db in shared_context["database_id_map"]:
            if db["name"].lower() == target_db_name.lower():
                logger.info(f"Found matching database: '{db['name']}' -> {db['id']}")
                return db["id"]
        
        logger.warning(f"Database '{target_db_name}' not found in existing mappings")
        return None
    
    # Need to fetch database list first
    logger.info("No database mappings in context, fetching database list...")
    
    if "list_notion_databases" not in tool_map:
        logger.error("list_notion_databases tool not available")
        return None
    
    try:
        list_tool = tool_map["list_notion_databases"]
        result = list_tool.run({})
        
        if isinstance(result, dict) and "databases" in result:
            # Store mappings in shared context
            shared_context["database_id_map"] = result["databases"]
            shared_context["database_name_map"] = {}
            
            for db in result["databases"]:
                db_name = db.get("name", "Untitled")
                db_id = db.get("id", "")
                shared_context["database_name_map"][db_name] = db_id
                logger.info(f"Mapped database '{db_name}' -> {db_id}")
                
                # Check if this is our target database
                if db_name.lower() == target_db_name.lower():
                    logger.info(f"Found target database: '{db_name}' -> {db_id}")
                    return db_id
            
            logger.warning(f"Database '{target_db_name}' not found in fetched list")
            logger.info(f"Available databases: {[db['name'] for db in result['databases']]}")
            return None
        else:
            logger.error(f"Failed to fetch database list: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching database list: {str(e)}")
        return None

def extract_parameters_from_query(user_query: str, task_description: str) -> Dict[str, Any]:
    """Extract parameters from user query and task description with enhanced logging"""
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
        params["path"] = file_match.group(1) or file_match.group(2)
        logger.info(f"Extracted path: {params['path']}")
    
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
        params["message"] = message_match.group(2)
        logger.info(f"Extracted commit message: {params['message']}")
    
    # Extract comments
    if "comment" in task_description.lower():
        comment_match = re.search(r'saying\s*["\']([^"\']+)[\'"]', user_query)
        if comment_match:
            params["comment"] = comment_match.group(1)
            logger.info(f"Extracted comment: {params['comment']}")
    
    # ENHANCED: Database name extraction for Notion operations
    if "database" in task_description.lower() or "database" in user_query.lower():
        logger.info("Database operation detected, extracting database information...")
        
        db_name = extract_database_name_from_query(user_query, task_description)
        if db_name:
            params["database_name"] = db_name
            logger.info(f"Extracted database_name: {params['database_name']}")
        
        # Extract priority information
        priority_patterns = [
            r"priority\s+['\"]([^'\"]+)['\"]",  # priority 'Medium'
            r"with\s+priority\s+['\"]([^'\"]+)['\"]",  # with priority 'Medium'
            r"items\s+with\s+priority\s+['\"]([^'\"]+)['\"]"  # items with priority 'Medium'
        ]
        
        for pattern in priority_patterns:
            priority_match = re.search(pattern, user_query, re.IGNORECASE)
            if priority_match:
                params["priority"] = priority_match.group(1)
                logger.info(f"Extracted priority: {params['priority']}")
                break
        
        # CRITICAL FIX: Enhanced database ID extraction for UUID format
        uuid_match = re.search(r'(?:with\s+ID|ID\s*[:=])\s*[\'"]?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})[\'"]?', user_query, re.IGNORECASE)
        if uuid_match:
            params["database_id"] = uuid_match.group(1)
            logger.info(f"Extracted database_id (UUID): {params['database_id']}")
        else:
            # Try to find any UUID in the query
            uuid_match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', user_query)
            if uuid_match:
                params["database_id"] = uuid_match.group(0)
                logger.info(f"Extracted database_id (found UUID): {params['database_id']}")
    
    # NEW: Detect if user wants database entries vs just schema
    entry_keywords = ["show entries", "list items", "database contents", "pages in", "records", "show data", "entries in"]
    schema_keywords = ["database details", "database schema", "database structure", "properties of"]

    if any(keyword in user_query.lower() for keyword in entry_keywords):
        params["query_entries"] = True
    elif any(keyword in user_query.lower() for keyword in schema_keywords):
        params["query_entries"] = False
    elif "details of" in user_query.lower():
        # Default behavior for "Get details of database X" - show both schema AND entries
        params["query_entries"] = True
        params["include_schema"] = True
        
    logger.info(f"Final extracted parameters: {params}")
    logger.info(f"=== PARAMETER EXTRACTION END ===")
    return params

def update_shared_context(shared_context: Dict[str, Any], tool_name: str, result: Any):
    """Update shared context with relevant information from tool execution"""
    logger.info(f"Updating shared context for tool: {tool_name}")
    
    if tool_name == "get_issue" and isinstance(result, dict):
        shared_context["issue_number"] = result.get("number")
        logger.info(f"Updated issue_number in context: {result.get('number')}")
    elif tool_name == "get_pull_request" and isinstance(result, dict):
        shared_context["pr_number"] = result.get("number")
        logger.info(f"Updated pr_number in context: {result.get('number')}")
    elif tool_name == "create_file" and isinstance(result, dict):
        if "content" in result and "path" in result["content"]:
            shared_context["last_created_file"] = result["content"]["path"]
            logger.info(f"Updated last_created_file in context: {result['content']['path']}")
    
    return shared_context

def task_executor(state: AgentState, tool_map: Dict[str, BaseTool], env_repo_url: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced task executor with better parameter handling and proper repository configuration"""
    logger.info("=== TASK EXECUTOR START ===")
    logger.info(f"Current state keys: {list(state.keys())}")
    
    task_queue = state["task_queue"]
    current_index = state["current_task_index"]
    task_results = state["task_results"].copy()
    shared_context = state["shared_context"].copy()
    execution_status = state["execution_status"].copy()
    errors = state["errors"].copy()
    
    logger.info(f"Current task index: {current_index}, Total tasks: {len(task_queue)}")
    logger.info(f"Current shared_context: {shared_context}")
    
    if current_index >= len(task_queue):
        logger.info("All tasks completed")
        return {
            **state,
            "is_complete": True
        }
    
    current_task = task_queue[current_index]
    task_id = current_task["task_id"]
    tool_name = current_task["tool_name"]
    service = current_task.get("service", "github")  # Default to github for backward compatibility
    
    logger.info(f"=== EXECUTING TASK {current_index + 1}/{len(task_queue)} ===")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Tool name: {tool_name}")
    logger.info(f"Service: {service}")
    logger.info(f"Task details: {current_task}")
    
    # NORMALIZE TOOL NAME TO HANDLE COMMON VARIATIONS
    tool_name_normalized = tool_name.lower()
    # Handle common naming variations
    if tool_name_normalized == "search_issues":
        tool_name_normalized = "search_issues_and_prs"
    elif tool_name_normalized in ["get_file_contents", "get_file", "readfile"]:
        tool_name_normalized = "read_file"
    elif tool_name_normalized in ["get_pull_request_files", "pr_files", "changed_files"]:
        tool_name_normalized = "list_pull_request_files"
    elif tool_name_normalized == "list_open_prs":
        tool_name_normalized = "list_open_pull_requests"
    
    logger.info(f"Normalized tool name: {tool_name} -> {tool_name_normalized}")
    
    try:
        # Mark task as in progress
        execution_status[task_id] = TaskStatus.IN_PROGRESS.value
        logger.info(f"Marked task {task_id} as IN_PROGRESS")
        
        # Get repository context for GitHub tasks
        repo_name = None
        if service == "github" and "repo_owner" in shared_context and "repo_name" in shared_context:
            repo_name = f"{shared_context['repo_owner']}/{shared_context['repo_name']}"
            logger.info(f"Using repository context: {repo_name}")
        
        # Get the tool instance
        if tool_name_normalized not in tool_map:
            error_msg = f"Tool {tool_name_normalized} not found in tool_map. Available tools: {list(tool_map.keys())}"
            logger.error(error_msg)
            execution_status[task_id] = TaskStatus.FAILED.value
            return {
                **state,
                "execution_status": execution_status,
                "errors": errors + [error_msg],
                "current_task_index": current_index + 1
            }
        
        # CRITICAL: Create a new tool instance with the correct repository context for GitHub tools
        if service == "github" and repo_name:
            tool = tool_map[tool_name_normalized].__class__(
                github_api_wrapper=get_wrapper(repo=repo_name)
            )
            logger.info(f"Created GitHub tool instance for repo: {repo_name}")
        else:
            tool = tool_map[tool_name_normalized]
            logger.info(f"Using existing tool instance for service: {service}")
        
        
        #ENHANCED: Database resolution for Notion tools that need database_id
        if service == "notion" and tool_name_normalized in ["get_notion_database", "query_notion_database"]:
            logger.info("=== NOTION DATABASE OPERATION - CHECKING FOR DATABASE RESOLUTION ===")
            
            # Extract database name from the query
            user_query = state.get("user_query", "")
            task_description = current_task.get("task_description", "")
            target_db_name = extract_database_name_from_query(user_query, task_description)
            
            if target_db_name:
                logger.info(f"Target database name: '{target_db_name}'")
                
                # Check if we already have the database ID in shared context
                database_id = None
                
                # Method 1: Check database_id_map
                if "database_id_map" in shared_context and shared_context["database_id_map"]:
                    logger.info("Checking existing database_id_map...")
                    for db in shared_context["database_id_map"]:
                        if db["name"].lower() == target_db_name.lower():
                            database_id = db["id"]
                            logger.info(f"Found matching database ID: {database_id}")
                            break
                
                # Method 2: Check database_name_map
                if not database_id and "database_name_map" in shared_context:
                    logger.info("Checking existing database_name_map...")
                    for name, db_id in shared_context["database_name_map"].items():
                        if name.lower() == target_db_name.lower():
                            database_id = db_id
                            logger.info(f"Found matching database ID: {database_id}")
                            break
                
                # Method 3: Need to fetch database list first
                if not database_id:
                    logger.info("Database ID not found in context, fetching database list...")
                    
                    if "list_notion_databases" in tool_map:
                        try:
                            list_tool = tool_map["list_notion_databases"]
                            list_result = list_tool.run({})
                            
                            if isinstance(list_result, dict) and "databases" in list_result:
                                # Store mappings in shared context
                                shared_context["database_id_map"] = list_result["databases"]
                                shared_context["database_name_map"] = {}
                                
                                for db in list_result["databases"]:
                                    db_name = db.get("name", "Untitled")
                                    db_id = db.get("id", "")
                                    shared_context["database_name_map"][db_name] = db_id
                                    logger.info(f"Mapped database '{db_name}' -> {db_id}")
                                    
                                    # Check if this is our target database
                                    if db_name.lower() == target_db_name.lower():
                                        database_id = db_id
                                        logger.info(f"Found target database: '{db_name}' -> {db_id}")
                                        break
                                
                                if not database_id:
                                    available_dbs = [db['name'] for db in list_result['databases']]
                                    error_msg = f"Database '{target_db_name}' not found. Available databases: {available_dbs}"
                                    logger.error(error_msg)
                                    execution_status[task_id] = TaskStatus.FAILED.value
                                    return {
                                        **state,
                                        "errors": errors + [error_msg],
                                        "execution_status": execution_status,
                                        "current_task_index": current_index + 1
                                    }
                            else:
                                error_msg = f"Failed to fetch database list: {list_result}"
                                logger.error(error_msg)
                                execution_status[task_id] = TaskStatus.FAILED.value
                                return {
                                    **state,
                                    "errors": errors + [error_msg],
                                    "execution_status": execution_status,
                                    "current_task_index": current_index + 1
                                }
                        except Exception as e:
                            error_msg = f"Error fetching database list: {str(e)}"
                            logger.error(error_msg)
                            execution_status[task_id] = TaskStatus.FAILED.value
                            return {
                                **state,
                                "errors": errors + [error_msg],
                                "execution_status": execution_status,
                                "current_task_index": current_index + 1
                            }
                    else:
                        error_msg = "list_notion_databases tool not available for database resolution"
                        logger.error(error_msg)
                        execution_status[task_id] = TaskStatus.FAILED.value
                        return {
                            **state,
                            "errors": errors + [error_msg],
                            "execution_status": execution_status,
                            "current_task_index": current_index + 1
                        }
                
                # Store the resolved database_id in shared context
                if database_id:
                    shared_context["database_id"] = database_id
                    logger.info(f"Set database_id in shared context: {database_id}")
            else:
                logger.error("Could not extract database name from query or task description")
        
        # Extract parameters from user query and shared context
        params = extract_parameters_from_query(
            state.get("user_query", ""), 
            current_task.get("task_description", "")
        )
        logger.info(f"Parameters after extraction: {params}")
        
        # Add shared context to parameters
        if service == "github":
            if "repo_owner" in shared_context and shared_context["repo_owner"]:
                params["owner"] = shared_context["repo_owner"]
                logger.info(f"Added owner from shared context: {params['owner']}")
            if "repo_name" in shared_context and shared_context["repo_name"]:
                params["repo"] = shared_context["repo_name"]
                logger.info(f"Added repo from shared context: {params['repo']}")
            
            # Add shared context parameters
            if "issue_number" in shared_context:
                params["issue_number"] = shared_context["issue_number"]
                logger.info(f"Added issue_number from shared context: {params['issue_number']}")
            if "pr_number" in shared_context:
                params["pr_number"] = shared_context["pr_number"]
                logger.info(f"Added pr_number from shared context: {params['pr_number']}")
        
        elif service == "notion":
            # Add database_id from shared context if available
            if "database_id" in shared_context:
                params["database_id"] = shared_context["database_id"]
                logger.info(f"Added database_id from shared context: {params['database_id']}")
        
        # Apply parameter name mapping if available for this tool
        if tool_name_normalized in PARAM_NAME_MAPPING:
            tool_mapping = PARAM_NAME_MAPPING[tool_name_normalized]
            mapped_params = params.copy()
            logger.info(f"Applying parameter mapping for {tool_name_normalized}: {tool_mapping}")
            
            for source_name, target_name in tool_mapping.items():
                if source_name in mapped_params and target_name not in mapped_params:
                    mapped_params[target_name] = mapped_params[source_name]
                    del mapped_params[source_name]
                    logger.info(f"Mapped parameter: {source_name} -> {target_name} = {mapped_params[target_name]}")
            
            params = mapped_params
            logger.info(f"Parameters after mapping: {params}")
        
        # Execute the tool
        logger.info("=== TOOL EXECUTION ===")
        try:
            if hasattr(tool.args_schema, 'model_fields') and tool.args_schema.model_fields:
                logger.info(f"Tool has parameters. Schema fields: {list(tool.args_schema.model_fields.keys())}")
                
                # Tool has parameters - only pass relevant ones
                filtered_params = {}
                for field_name, field_info in tool.args_schema.model_fields.items():
                    if field_name in params:
                        filtered_params[field_name] = params[field_name]
                        logger.info(f"Added parameter {field_name}: {params[field_name]}")
                    else:
                        # Use default value if available
                        if field_info.default is not None and field_info.default != PydanticUndefined:
                            filtered_params[field_name] = field_info.default
                            logger.info(f"Using default value for '{field_name}': {field_info.default}")
                        elif field_info.is_required():
                            logger.error(f"Required parameter '{field_name}' not found for {tool_name_normalized}")
                            logger.error(f"Available parameters: {params}")
                
                # Ensure we have all required parameters
                required_fields = [f for f, v in tool.args_schema.model_fields.items() if v.is_required()]
                missing_fields = [f for f in required_fields if f not in filtered_params]
                if missing_fields:
                    error_msg = f"Missing required parameters for {tool_name_normalized}: {missing_fields}"
                    logger.error(error_msg)
                    logger.error(f"Available parameters: {params}")
                    logger.error(f"Filtered parameters: {filtered_params}")
                    execution_status[task_id] = TaskStatus.FAILED.value
                    return {
                        **state,
                        "errors": errors + [error_msg],
                        "execution_status": execution_status,
                        "current_task_index": current_index + 1
                    }
                
                logger.info(f"Executing tool with parameters: {filtered_params}")
                result = tool.run(filtered_params)
            else:
                # Tool has no parameters
                logger.info("Tool has no parameters, executing with empty dict")
                result = tool.run({})
            
            logger.info(f"Tool execution completed successfully")
            logger.info(f"Result type: {type(result)}")
            if isinstance(result, dict) and len(str(result)) < 500:
                logger.info(f"Result: {result}")
            else:
                logger.info(f"Result summary: {str(result)[:200]}...")
            
            # Store result
            task_results[task_id] = result
            execution_status[task_id] = TaskStatus.COMPLETED.value
            logger.info(f"Marked task {task_id} as COMPLETED")
            
            # Update shared context with relevant information
            if service == "github":
                update_shared_context(shared_context, tool_name_normalized, result)
            
            # After successful get_notion_database execution:
            if (service == "notion" and tool_name_normalized == "get_notion_database" and "query_entries" in params and params.get("query_entries") and execution_status.get(task_id) == TaskStatus.COMPLETED.value):
                
                logger.info("Database schema retrieved, now querying entries...")
                
                # Create a follow-up task to query database entries
                follow_up_task = {
                    "task_id": f"notion_query_entries_{current_index + 1}",
                    "tool_name": "query_notion_database",
                    "service": "notion",
                    "task_description": f"Query entries from database {shared_context.get('database_id')}",
                    "auto_generated": True
                }
                
                # Insert the task right after current task
                task_queue.insert(current_index + 1, follow_up_task)
                shared_context["auto_task_added"] = True
                
                logger.info(f"Added follow-up task to query database entries")
                
            elif service == "notion" and tool_name_normalized == "list_notion_databases":
                # Process results to build database name-to-ID mapping
                logger.info("=== PROCESSING NOTION DATABASE LISTING RESULTS ===")
                if isinstance(result, dict) and "databases" in result:
                    # Create or update database_id_map in shared context
                    shared_context["database_id_map"] = result["databases"]
                    shared_context["database_name_map"] = {}
                    
                    for db in result["databases"]:
                        db_name = db.get("name", "Untitled")
                        db_id = db.get("id", "")
                        
                        shared_context["database_name_map"][db_name] = db_id
                        logger.info(f"Mapped database '{db_name}' to ID: {db_id}")
                    
                    logger.info(f"Updated database mappings in shared context")
                    
                    # If we have a pending database resolution, try to resolve it now
                    if "pending_db_resolution" in shared_context:
                        pending_db_name = shared_context["pending_db_resolution"]
                        logger.info(f"Resolving pending database: '{pending_db_name}'")
                        
                        for db in result["databases"]:
                            if db["name"].lower() == pending_db_name.lower():
                                shared_context["database_id"] = db["id"]
                                logger.info(f"Resolved pending database '{pending_db_name}' -> {db['id']}")
                                del shared_context["pending_db_resolution"]
                                break
                            # If we couldn't find the database, that's an error
                            if "database_id" not in shared_context:
                                available_dbs = [db['name'] for db in result['databases']]
                                error_msg = f"Database '{pending_db_name}' not found. Available databases: {available_dbs}"
                                logger.error(error_msg)
                                execution_status[task_id] = TaskStatus.FAILED.value
                                return {
                                    **state,
                                    "errors": errors + [error_msg],
                                    "execution_status": execution_status,
                                    "current_task_index": current_index + 1,
                                    "shared_context": shared_context
                                }
                                
            logger.info(f"=== TASK {task_id} COMPLETED SUCCESSFULLY ===")
            
        except Exception as tool_error:
            error_msg = f"Tool execution failed for {tool_name}: {str(tool_error)}"
            execution_status[task_id] = TaskStatus.FAILED.value
            errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Tool error details: {type(tool_error).__name__}: {tool_error}")
        
        return {
            **state,
            "current_task_index": current_index + 1,
            "task_results": task_results,
            "shared_context": shared_context,
            "execution_status": execution_status,
            "errors": errors
        }
        
    except Exception as e:
        error_msg = f"Task executor failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception details: {type(e).__name__}: {e}")
        execution_status[task_id] = TaskStatus.FAILED.value
        return {
            **state,
            "errors": errors + [error_msg],
            "current_task_index": current_index + 1,
            "execution_status": execution_status
        }