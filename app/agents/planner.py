#planner.py
import os
import json
import logging
import re
from typing import List,Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.utils import AgentState
from app.tools.github_tools import get_github_tools, get_wrapper

logger = logging.getLogger(__name__)

def check_github_rate_limit():
    """Check GitHub API rate limit status"""
    try:
        wrapper = get_wrapper()
        rate_limit = wrapper.github.get_rate_limit()
        return rate_limit.core.remaining
    except:
        return 0

def extract_github_info(state: AgentState, env_repo_url: Optional[str] = None) -> Dict[str, str]:
    """Extract GitHub repository information with fallback to state values"""
    repo_info = {
        "repo_owner": state.get("repo_owner", ""),
        "repo_name": state.get("repo_name", ""),
        "github_repo_url": state.get("github_repo_url", "")
    }
    
    query = state["user_query"]
    
    # First, try to extract from URL pattern
    url_pattern = r'https?://github\.com/([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_\.]+)'
    match = re.search(url_pattern, query)
    if match:
        owner = match.group(1)
        repo = match.group(2).rstrip('.git?#')
        repo_info["repo_owner"] = owner
        repo_info["repo_name"] = repo
        repo_info["github_repo_url"] = f"https://github.com/{owner}/{repo}".strip()
    
    # If no URL found, try to extract owner/repo pattern
    if not repo_info["repo_owner"] or not repo_info["repo_name"]:
        owner_repo_pattern = r'([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_\.]+)'
        match = re.search(owner_repo_pattern, query)
        if match:
            owner = match.group(1)
            repo = match.group(2).rstrip('.git?#')
            repo_info["repo_owner"] = owner
            repo_info["repo_name"] = repo
            repo_info["github_repo_url"] = f"https://github.com/{owner}/{repo}"
    
    # If still no repo info and we have an env_repo_url, use that
    if (not repo_info["repo_owner"] or not repo_info["repo_name"]) and env_repo_url:
        # Check if it's a full URL
        url_match = re.search(r'https?://github\.com/([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_\.]+)', env_repo_url)
        if url_match:
            owner = url_match.group(1)
            repo = url_match.group(2).rstrip('.git?#')
            repo_info["repo_owner"] = owner
            repo_info["repo_name"] = repo
            repo_info["github_repo_url"] = env_repo_url
        else:
            # Check if it's in owner/repo format (like your sba-workspace/leetcode)
            owner_repo_match = re.search(r'([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_\.]+)', env_repo_url)
            if owner_repo_match:
                owner = owner_repo_match.group(1)
                repo = owner_repo_match.group(2).rstrip('.git?#')
                repo_info["repo_owner"] = owner
                repo_info["repo_name"] = repo
                repo_info["github_repo_url"] = f"https://github.com/{owner}/{repo}"
    
    # CRITICAL: Ensure we have valid values
    if not repo_info["repo_owner"] or not repo_info["repo_name"]:
        logger.warning("Could not extract repository info, using state values as fallback")
        repo_info["repo_owner"] = state.get("repo_owner", "")
        repo_info["repo_name"] = state.get("repo_name", "")
        repo_info["github_repo_url"] = state.get("github_repo_url", "")
    
    # FINAL CHECK: If still empty, log error but return empty strings (not empty dict)
    if not repo_info["repo_owner"] or not repo_info["repo_name"]:
        logger.error("No valid repository information could be determined from query or environment")
    
    return repo_info

def parse_planner_response(response_content: str) -> Dict:
    """Robustly parse the planner's JSON response with multiple fallbacks"""
    # Clean the response
    clean_response = response_content.strip()
    # Case 1: Response is pure JSON
    if clean_response.startswith('{') and clean_response.endswith('}'):
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            pass
    # Case 2: Response has JSON inside markdown code block
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', clean_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Case 3: Extract any JSON object from the text
    json_match = re.search(r'\{[\s\S]*\}', clean_response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    # Case 4: Fix common JSON issues (single quotes, etc.)
    fixed_response = clean_response.replace("'", '"')
    if fixed_response.startswith('{') and fixed_response.endswith('}'):
        try:
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            pass
    # Case 5: Handle fragmented responses like '"github"'
    if '"github"' in clean_response:
        # Try to build a valid structure from fragments
        github_match = re.search(r'"github"\s*:\s*(\{.*?\})', clean_response)
        if github_match:
            try:
                return {"github": json.loads(github_match.group(1))}
            except json.JSONDecodeError:
                pass
    # Final fallback: return empty valid structure
    logger.warning(f"Could not parse planner response, using fallback: {clean_response}")
    return {"github": {}}

def validate_execution_plan(plan: Dict) -> bool:
    """Validate that the execution plan matches the required structure for ANY service"""
    if not isinstance(plan, dict):
        return False
    
    # Check for either github or notion section
    has_github = "github" in plan and isinstance(plan["github"], dict)
    has_notion = "notion" in plan and isinstance(plan["notion"], dict)
    
    if not (has_github or has_notion):
        return False
    
    # Validate GitHub section if present
    if has_github:
        for tool_name, tasks in plan["github"].items():
            if not isinstance(tasks, dict):
                return False
            for task_key, task_desc in tasks.items():
                if not isinstance(task_key, str) or not isinstance(task_desc, str):
                    return False
    
    # Validate Notion section if present
    if has_notion:
        for tool_name, tasks in plan["notion"].items():
            if not isinstance(tasks, dict):
                return False
            for task_key, task_desc in tasks.items():
                if not isinstance(task_key, str) or not isinstance(task_desc, str):
                    return False
    
    return True

def planner_agent(state: AgentState, llm: ChatGoogleGenerativeAI, env_repo_url: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced planner agent using the context prompt structure"""
    logger.info("Starting enhanced planner agent")
    # Extract GitHub repository information
    github_info = extract_github_info(state, env_repo_url)
    
    # CRITICAL: Check if we have valid repository info
    if not github_info["repo_owner"] or not github_info["repo_name"]:
        error_msg = "Repository owner and name could not be determined. Please specify the repository in your query or ensure GITHUB_REPO_URL is properly set."
        logger.error(error_msg)
        return {
            **state,
            "errors": state.get("errors", []) + [error_msg],
            "is_complete": True,
            "execution_plan": {"github": {}},
            "task_queue": []
        }
    
    # ADD THE ERROR CHECKING CODE HERE - RIGHT AFTER REPO VALIDATION
    # Check for specific GitHub API errors from previous steps
    if state.get("errors"):
        for error in state["errors"]:
            error_str = str(error).lower()
            if "rate limit" in error_str or "403" in error_str:
                remaining = check_github_rate_limit()
                wait_time = "a few minutes" if remaining < 10 else "a moment"
                return {
                    **state,
                    "errors": ["GitHub API rate limit exceeded. Please wait a few minutes and try again."],
                    "is_complete": True,
                    "execution_plan": {"github": {}},
                    "task_queue": []
                }
            if "repository not found" in error_str or "404" in error_str:
                return {
                    **state,
                    "errors": ["Repository not found. Please check the repository name and your permissions."],
                    "is_complete": True,
                    "execution_plan": {"github": {}},
                    "task_queue": []
                }
            if "resource not accessible by integration" in error_str:
                return {
                    **state,
                    "errors": ["Permission denied: The GitHub App does not have permission to access this repository."],
                    "is_complete": True,
                    "execution_plan": {"github": {}},
                    "task_queue": []
                }
                
                
    owner = github_info["repo_owner"]
    repo = github_info["repo_name"]
    
    #repository context to the user query for better planning
    augmented_query = f"{state['user_query']} (repository: {owner}/{repo})"
    
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a planner agent for GitHub and Notion operations. You MUST follow these instructions EXACTLY:
        ## STRICT OUTPUT REQUIREMENTS
        - OUTPUT ONLY A VALID JSON OBJECT - NO ADDITIONAL TEXT
        - THE JSON MUST HAVE SERVICE KEYS CONTAINING A DICTIONARY OF TOOLS
        - SUPPORTED SERVICE KEYS: "github", "notion"
        - EACH SERVICE DICTIONARY CONTAINS TOOL NAMES AS KEYS
        - YOU CAN USE MULTIPLE TOOLS PER SERVICE PER QUERY
        - USE ONLY THESE EXACT TOOL NAMES (CASE-SENSITIVE):
        
        GITHUB TOOLS:
            * get_issues
            * get_issue
            * comment_on_issue
            * search_issues_and_prs
            * list_open_pull_requests
            * get_pull_request
            * create_pull_request
            * list_pull_request_files
            * create_file
            * read_file
            * update_file
            * delete_file
            * overview_files_main_branch
            * overview_files_current_branch
            
        NOTION TOOLS:
            * list_notion_databases
            * get_notion_database
            * query_notion_database
            * create_notion_page
            * get_notion_page
            * update_notion_page
            * search_notion
        
        ## CRITICAL FORMAT RULES FOR BOTH SERVICES
        - Task descriptions must be NATURAL LANGUAGE descriptions of what needs to be done
        - DO NOT include parameter values in the task description
        - Parameters will be extracted by the executor based on the description
        - Each task must have a unique task key (e.g., "task_1", "task_2")
        
        ## SPECIAL INSTRUCTIONS FOR search_issues_and_prs:
        - The description MUST be a valid GitHub search query in GitHub's search syntax
        - Examples:
            * "is:pr is:closed" - to find closed pull requests
            * "is:issue is:open author:username" - to find open issues by a user
            * "repo:owner/repo is:pr is:open" - to find open PRs in a specific repo
        - DO NOT use natural language like "List all closed pull requests"
        - CONVERT the user's request into the appropriate GitHub search syntax
        
        ## SPECIAL INSTRUCTIONS FOR list_notion_databases:
        - The description MUST include the search term when filtering databases
        - Example: "Find databases related to interview"
        - Format as: "list_notion_databases: query=interview"
        
        ## SPECIAL INSTRUCTIONS FOR QUERYING BY DATABASE NAME:
        - If the user refers to a database by name (e.g., "Products"), you MUST create a two-step plan:
        1. First use list_notion_databases with the name as query parameter
        2. Then use query_notion_database with the found database ID
        - NEVER try to query a database using its name as the ID
        - Example for "Query Products database for Medium priority items":
        {{"notion": {{"list_notion_databases": {{"task_1": "Find database named Products"}}, "query_notion_database": {{"task_2": "database_id={{FOUND_ID}}, filter={{'property': 'Priority', 'select': {{'equals': 'Medium'}}}}"}}}}}}
        
        ## SPECIAL INSTRUCTIONS FOR DATABASE DETAILS:
        - When user asks "Get details of database X", create a plan that gets BOTH schema AND entries:
        - Use get_notion_database for schema, then query_notion_database for entries
        - Example: {{"notion": {{"get_notion_database": {{"task_1": "Get schema of DP Interview Prep database"}}, "query_notion_database": {{"task_2": "Get entries from DP Interview Prep database"}}}}}}
                
        ## CORRECT EXAMPLES FOR GITHUB (COPY THIS FORMAT EXACTLY):
        {{"github": {{"get_issue": {{"task_1": "Get issue #3275 from microsoft/vscode"}}}}}}
        {{"github": {{"search_issues_and_prs": {{"task_1": "is:pr is:closed repo:QwenLM/Qwen"}}}}}}
        
        ## CORRECT EXAMPLES FOR NOTION (COPY THIS FORMAT EXACTLY):
        {{"notion": {{"list_notion_databases": {{"task_1": "Find databases related to interview"}}}}}}
        {{"notion": {{"query_notion_database": {{"task_1": "database_id=abc123, filter={{'property': 'Status', 'select': {{'equals': 'Completed'}}}}"}}}}}}
        {{"notion": {{"get_notion_database": {{"task_1": "Get details of the database with ID abc123"}}}}}}
        {{"notion": {{"query_notion_database": {{"task_1": "Query the Tasks database for open items assigned to me"}}}}}}
        {{"notion": {{"query_notion_database": {{"task_1": "Query the Projects database for items with status In Progress"}}}}}}
        {{"notion": {{"get_notion_page": {{"task_1": "Get details of the page with ID def456"}}}}}}
        {{"notion": {{"update_notion_page": {{"task_1": "Update the page with ID def456 to set the status to Completed"}}}}}}
        {{"notion": {{"create_notion_page": {{"task_1": "Create a new page in the Tasks database with title Review documentation and priority High"}}}}}}
        {{"notion": {{"search_notion": {{"task_1": "Search for pages about Q3 planning"}}}}}}
        {{"notion": {{"search_notion": {{"task_1": "Find all pages mentioning budget in my workspace"}}}}}}
        
        ## INCORRECT EXAMPLES (NEVER DO THIS):
        - {{"github": "get_issue"}}
        - {{"github": ["get_issue"]}}
        - {{"github": {{"get_issue"}}}}
        - {{"github": {{"get_issue": "Get issue #3275"}}}}
        - {{"notion": "list_notion_databases"}}
        - {{"notion": ["list_notion_databases"]}}
        - {{"notion": {{"list_notion_databases"}}}}
        - {{"notion": {{"list_notion_databases": "List all databases"}}}}
        - {{"notion": {{"query_notion_database": {{"task_1": "database_id=abc123, filter={{'property': 'Status', 'select': {{'equals': 'Completed'}}}}"}}}}}}  // WRONG - don't put parameters in description
        
        ## VALIDATION RULES
        1. Must start with {{ and end with }}
        2. Must contain "github" OR "notion" as a key with an object value
        3. Each tool must map to a dictionary of tasks
        4. Each task must have a string description
        
        If you cannot create a valid plan, return: {{"notion": {{}}}} if Notion was requested, otherwise {{"github": {{}}}}

        NEVER output anything except a valid JSON object."""),
        ("human", "User Query: {query}\nRepository: {owner}/{repo}")
    ])
    
    try:
        response = llm.invoke(planner_prompt.format_messages(
            query=augmented_query,
            owner=owner,
            repo=repo
        ))
        logger.debug(f"Raw planner response: {response.content}")
        # Parse the execution plan
        plan_text = response.content.strip()
        
        # Handle common malformed responses
        if plan_text == '"github"':
            logger.warning("LLM returned invalid minimal response, using fallback structure")
            execution_plan = {"github": {}}
        elif plan_text.startswith('"') and plan_text.endswith('"'):
            logger.warning(f"LLM returned string instead of object: {plan_text}")
            execution_plan = {"github": {}}
        else:
            # Try to extract JSON from any wrapper text
            json_match = re.search(r'(\{[\s\S]*\})', plan_text)
            if json_match:
                try:
                    execution_plan = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse extracted JSON, using fallback")
                    execution_plan = {"github": {}}
            else:
                logger.warning("No JSON structure found, using fallback")
                execution_plan = {"github": {}}
        
        # CRITICAL FIX: Ensure proper nested structure
        if isinstance(execution_plan, dict):
            # Case 1: {"github": "tool_name"}
            if "github" in execution_plan and isinstance(execution_plan["github"], str):
                tool_name = execution_plan["github"]
                execution_plan = {
                    "github": {
                        tool_name: {
                            "default_task": f"Execute {tool_name} for the repository"
                        }
                    }
                }
                logger.warning(f"Fixed flat tool structure: converted '{tool_name}' to proper nested structure")
            # Case 2: {"github": {"tool_name"}}
            elif "github" in execution_plan and isinstance(execution_plan["github"], dict):
                fixed_tools = {}
                for tool_name, value in execution_plan["github"].items():
                    if not isinstance(value, dict):
                        # Convert {"tool_name": value} to {"tool_name": {"default_task": value}}
                        if isinstance(value, str):
                            fixed_tools[tool_name] = {"default_task": value}
                        else:
                            fixed_tools[tool_name] = {"default_task": f"Execute {tool_name}"}
                    else:
                        fixed_tools[tool_name] = value
                execution_plan["github"] = fixed_tools
        
        # Final validation
        if not isinstance(execution_plan, dict) or (not execution_plan.get("github") and not execution_plan.get("notion")):
            logger.warning("Execution plan missing valid service keys, creating default structure")
            # Use notion as fallback if the query is about Notion
            if any(word in state["user_query"].lower() for word in ["notion", "database", "page", "workspace"]):
                execution_plan = {"notion": {}}
            else:
                execution_plan = {"github": {}}
            
        valid_services = []
        for service in ["github", "notion"]:
            if service in execution_plan and isinstance(execution_plan[service], dict):
                valid_services.append(service)
                
        if not valid_services:
            logger.error("Execution plan contains no valid services (github/notion)")
            execution_plan = {"github": {}, "notion": {}}
        
        # Convert plan to task queue
        task_queue = create_task_queue(execution_plan)
        
        return {
            **state,
            "execution_plan": execution_plan,
            "task_queue": task_queue,
            "current_task_index": 0,
            "repo_owner": owner,
            "repo_name": repo,
            "github_repo_url": github_info["github_repo_url"],
            "task_results": {},
            "shared_context": {
                "repo_owner":owner,
                "repo_name": repo,
                "repo_url": github_info["github_repo_url"]
            },
            "execution_status": {},
            "is_complete": False,
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "errors": [],
            "messages": state.get("messages", []) + [response]
        }     
    except Exception as e:
        logger.exception(f"Planning failed completely: {str(e)}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Planning failed: {str(e)}"],
            "execution_plan": {"github": {}},
            "task_queue": [],
            "is_complete": True
        }

def create_task_queue(execution_plan: Dict) -> List[Dict[str, Any]]:
    """Convert execution plan to a flat task queue with robust error handling"""
    task_queue = []
    
    # Process GitHub section
    github_section = execution_plan.get("github", {})
    if not isinstance(github_section, dict):
        logger.warning(f"GitHub section is not a dictionary: {type(github_section)}")
        github_section = {}
    for service in ["github", "notion"]:
        section = execution_plan.get(service, {})
        if not isinstance(section, dict):
            continue
    for tool_name, tasks in github_section.items():
        # Skip if tool name isn't a string
        if not isinstance(tool_name, str):
            continue
        # Handle cases where tasks might not be a dict
        if not isinstance(tasks, dict):
            if isinstance(tasks, str) and tasks.strip():
                # If it's a string, treat it as a single task
                task_queue.append({
                    "service": "github",
                    "tool_name": tool_name,
                    "task_key": "default_task",
                    "task_description": tasks,
                    "task_id": f"github_{tool_name}_default_task"
                })
            continue
        
        for task_key, task_description in tasks.items():
            # Skip invalid task descriptions
            if not task_description or not isinstance(task_description, str):
                continue
            
            task_queue.append({
                "service": "github",
                "tool_name": tool_name,
                "task_key": task_key,
                "task_description": task_description,
                "task_id": f"github_{tool_name}_{task_key}"
            })
    
    # Process Notion section
    notion_section = execution_plan.get("notion", {})
    
    if not isinstance(notion_section, dict):
        logger.warning(f"Notion section is not a dictionary: {type(notion_section)}")
        notion_section = {}
    
    for tool_name, tasks in notion_section.items():
        # Skip if tool name isn't a string
        if not isinstance(tool_name, str):
            continue
        # Handle cases where tasks might not be a dict
        if not isinstance(tasks, dict):
            if isinstance(tasks, str) and tasks.strip():
                # If it's a string, treat it as a single task
                task_queue.append({
                    "service": "notion",
                    "tool_name": tool_name,
                    "task_key": "default_task",
                    "task_description": tasks,
                    "task_id": f"notion_{tool_name}_default_task"
                })
            continue
        
        for task_key, task_description in tasks.items():
            # Skip invalid task descriptions
            if not task_description or not isinstance(task_description, str):
                continue
            
            task_queue.append({
                "service": "notion",
                "tool_name": tool_name,
                "task_key": task_key,
                "task_description": task_description,
                "task_id": f"notion_{tool_name}_{task_key}"
            })
    
    # Log what tasks we're creating
    if task_queue:
        logger.info(f"Created {len(task_queue)} tasks from plan ({len([t for t in task_queue if t['service'] == 'github'])} GitHub, {len([t for t in task_queue if t['service'] == 'notion'])} Notion)")
        for i, task in enumerate(task_queue):
            logger.debug(f"Task {i+1}: {task['service']}.{task['tool_name']} - {task['task_key']}: {task['task_description'][:50]}...")
    else:
        logger.warning("No tasks created from execution plan")
    
    return task_queue