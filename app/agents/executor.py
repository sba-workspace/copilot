import os
import logging
import re
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
    }
}

def extract_parameters_from_query(user_query: str, task_description: str) -> Dict[str, Any]:
    """Extract parameters from user query and task description"""
    params = {}
    # Extract issue numbers
    issue_match = re.search(r'issue\s*(?:number\s*)?#?(\d+)', user_query.lower())
    if issue_match:
        params["issue_number"] = int(issue_match.group(1))
    # Extract PR numbers
    pr_match = re.search(r'(?:pr|pull\s*request)\s*(?:number\s*)?#?(\d+)', user_query.lower())
    if pr_match:
        params["pr_number"] = int(pr_match.group(1))
    # Extract file paths
    file_match = re.search(r'(?:file\s*(?:called\s*)?["\']([^"\']+)[\'"]|([a-zA-Z0-9_.-]+\.[a-zA-Z]+))', user_query)
    if file_match:
        params["path"] = file_match.group(1) or file_match.group(2)
    # Extract search queries
    if "search" in task_description.lower():
        search_match = re.search(r'["\']([^"\']+)[\'"]', user_query)
        if search_match:
            params["query"] = search_match.group(1)
    # Extract file content (handles multi-line content)
    content_match = re.search(r'with\s+content\s+(["\'])([\s\S]*?)\1', user_query)
    if content_match:
        params["content"] = content_match.group(2)
    # Extract commit message
    message_match = re.search(r'(?:with\s+message|commit\s+message)\s+(["\'])([\s\S]*?)\1', user_query)
    if message_match:
        params["message"] = message_match.group(2)
    # Extract comments
    if "comment" in task_description.lower():
        comment_match = re.search(r'saying\s*["\']([^"\']+)[\'"]', user_query)
        if comment_match:
            params["comment"] = comment_match.group(1)
    return params

def update_shared_context(shared_context: Dict[str, Any], tool_name: str, result: Any):
    """Update shared context with relevant information from tool execution"""
    if tool_name == "get_issue" and isinstance(result, dict):
        shared_context["issue_number"] = result.get("number")
    elif tool_name == "get_pull_request" and isinstance(result, dict):
        shared_context["pr_number"] = result.get("number")
    elif tool_name == "create_file" and isinstance(result, dict):
        if "content" in result and "path" in result["content"]:
            shared_context["last_created_file"] = result["content"]["path"]
    # Add more context updates as needed for different tools
    return shared_context

def task_executor(state: AgentState, tool_map: Dict[str, BaseTool], env_repo_url: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced task executor with better parameter handling and proper repository configuration"""
    logger.info("Starting enhanced task executor")
    task_queue = state["task_queue"]
    current_index = state["current_task_index"]
    task_results = state["task_results"].copy()
    shared_context = state["shared_context"].copy()
    execution_status = state["execution_status"].copy()
    errors = state["errors"].copy()
    
    if current_index >= len(task_queue):
        return {
            **state,
            "is_complete": True
        }
    
    current_task = task_queue[current_index]
    task_id = current_task["task_id"]
    tool_name = current_task["tool_name"]
    
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
    logger.info(f"Executing task: {task_id}")
    
    try:
        # Mark task as in progress
        execution_status[task_id] = TaskStatus.IN_PROGRESS.value
        
        # CRITICAL: Get repository context for this task
        repo_name = None
        if "repo_owner" in shared_context and "repo_name" in shared_context:
            repo_name = f"{shared_context['repo_owner']}/{shared_context['repo_name']}"
            logger.info(f"Using repository context: {repo_name}")
        
        # Get the tool instance
        if tool_name_normalized not in tool_map:
            error_msg = f"Tool {tool_name} not found"
            execution_status[task_id] = TaskStatus.FAILED.value
            return {
                **state,
                "execution_status": execution_status,
                "errors": errors + [error_msg],
                "current_task_index": current_index + 1
            }
        
        # CRITICAL: Create a new tool instance with the correct repository context
        if repo_name:
            tool = tool_map[tool_name_normalized].__class__(
                github_api_wrapper=get_wrapper(repo=repo_name)
            )
        else:
            tool = tool_map[tool_name_normalized]
        
        # Extract parameters from user query and shared context
        params = extract_parameters_from_query(
            state["user_query"], 
            current_task["task_description"]
        )
        logger.debug(f"Extracted parameters before mapping: {params}")
        
        # Add repository context to parameters (for tools that need it)
        if "repo_owner" in shared_context and shared_context["repo_owner"]:
            params["owner"] = shared_context["repo_owner"]
        if "repo_name" in shared_context and shared_context["repo_name"]:
            params["repo"] = shared_context["repo_name"]
        
        # Add shared context parameters
        if "issue_number" in shared_context:
            params["issue_number"] = shared_context["issue_number"]
        if "pr_number" in shared_context:
            params["pr_number"] = shared_context["pr_number"]
        
        # Apply parameter name mapping if available for this tool
        if tool_name_normalized in PARAM_NAME_MAPPING:
            tool_mapping = PARAM_NAME_MAPPING[tool_name_normalized]
            mapped_params = params.copy()
            for source_name, target_name in tool_mapping.items():
                if source_name in mapped_params and target_name not in mapped_params:
                    mapped_params[target_name] = mapped_params[source_name]
                    del mapped_params[source_name]
            params = mapped_params
            logger.debug(f"Parameters after mapping: {params}")
        
        # For commenting
        if tool_name_normalized == "comment_on_issue":
            if "comment" not in params:
                task_desc = current_task["task_description"]
                comment_match = re.search(r"Comment\s+'([^']+)'", task_desc)
                if comment_match:
                    params["comment"] = comment_match.group(1)
                else:
                    quoted_match = re.search(r"'([^']+)'", task_desc)
                    if quoted_match:
                        params["comment"] = quoted_match.group(1)
        
        # Add extraction for branch information for file operations
        if tool_name_normalized in ["read_file", "list_pull_request_files", "create_file", "update_file", "delete_file"]:
            # Extract branch from query if specified
            branch_match = re.search(r'from\s+branch\s+["\']?([a-zA-Z0-9\-_/.]+)["\']?', state["user_query"].lower())
            if branch_match:
                params["branch"] = branch_match.group(1)
            # Default to 'main' if no branch specified
            elif "branch" not in params:
                params["branch"] = "main"
        
         # For search issues and PRs
        if tool_name_normalized == "search_issues_and_prs":
            # Ensure the query parameter is present and properly formatted
            if "query" not in params:
                # Convert natural language to GitHub search syntax
                task_desc = current_task["task_description"].lower()
                repo_owner = shared_context.get("repo_owner", "")
                repo_name = shared_context.get("repo_name", "")
                
                if "closed" in task_desc and "pull request" in task_desc:
                    params["query"] = f"is:pr is:closed repo:{repo_owner}/{repo_name}"
                elif "open" in task_desc and "pull request" in task_desc:
                    params["query"] = f"is:pr is:open repo:{repo_owner}/{repo_name}"
                elif "issue" in task_desc and "closed" in task_desc:
                    params["query"] = f"is:issue is:closed repo:{repo_owner}/{repo_name}"
                elif "issue" in task_desc and "open" in task_desc:
                    params["query"] = f"is:issue is:open repo:{repo_owner}/{repo_name}"
                else:
                    # Default to searching in the current repository
                    params["query"] = f"repo:{repo_owner}/{repo_name} {task_desc}"
                logger.info(f"Generated GitHub search query: {params['query']}")
        
        
        # Ensure required parameters for file operations
        if tool_name_normalized in ["create_file", "update_file", "delete_file"]:
            # Set default commit message if not provided
            if "commit_message" not in params:
                file_path = params.get('file_path', 'file')
                file_name = os.path.basename(file_path)
                if tool_name_normalized == "create_file":
                    params["commit_message"] = f"Create {file_name}"
                elif tool_name_normalized == "update_file":
                    params["commit_message"] = f"Update {file_name}"
                elif tool_name_normalized == "delete_file":
                    params["commit_message"] = f"Delete {file_name}"
            
            # For create_file specifically, ensure content is provided
            if tool_name_normalized == "create_file" and "content" not in params:
                # Use a fallback message if content wasn't extracted
                params["content"] = "# File created by AI assistant"
        
        # Execute the tool
        try:
            if hasattr(tool.args_schema, 'model_fields') and tool.args_schema.model_fields:
                # Tool has parameters - only pass relevant ones
                filtered_params = {}
                for field_name in tool.args_schema.model_fields.keys():
                    if field_name in params:
                        filtered_params[field_name] = params[field_name]
                    else:
                        # Use default value if available
                        field = tool.args_schema.model_fields[field_name]
                        if field.default is not None and field.default != PydanticUndefined:
                            filtered_params[field_name] = field.default
                            logger.info(f"Using default value for '{field_name}': {field.default}")
                        else:
                            logger.warning(f"Required parameter '{field_name}' not found for {tool_name_normalized}")
                
                # Ensure we have all required parameters
                required_fields = [f for f, v in tool.args_schema.model_fields.items() if v.is_required()]
                missing_fields = [f for f in required_fields if f not in filtered_params]
                if missing_fields:
                    logger.error(f"Missing required parameters for {tool_name_normalized}: {missing_fields}")
                
                result = tool.run(filtered_params)
            else:
                # Tool has no parameters
                result = tool.run({})
            
            # Store result
            task_results[task_id] = result
            execution_status[task_id] = TaskStatus.COMPLETED.value
            
            # Update shared context with relevant information
            update_shared_context(shared_context, tool_name_normalized, result)
            
            logger.info(f"Task {task_id} completed successfully")
        except Exception as tool_error:
            error_msg = f"Tool execution failed for {tool_name}: {str(tool_error)}"
            execution_status[task_id] = TaskStatus.FAILED.value
            errors.append(error_msg)
            logger.error(error_msg)
        
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
        execution_status[task_id] = TaskStatus.FAILED.value
        return {
            **state,
            "errors": errors + [error_msg],
            "current_task_index": current_index + 1,
            "execution_status": execution_status
        }