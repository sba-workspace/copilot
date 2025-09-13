#planner_mcp.py - New MCP-based planner
import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.utils import AgentState

logger = logging.getLogger(__name__)

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
        repo_info["repo_owner"] = match.group(1)
        repo_info["repo_name"] = match.group(2)
        repo_info["github_repo_url"] = f"https://github.com/{match.group(1)}/{match.group(2)}"
    
    # If no URL found, try to extract owner/repo pattern
    if not repo_info["repo_owner"] or not repo_info["repo_name"]:
        owner_repo_pattern = r'([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_\.]+)'
        match = re.search(owner_repo_pattern, query)
        if match:
            repo_info["repo_owner"] = match.group(1)
            repo_info["repo_name"] = match.group(2)
            repo_info["github_repo_url"] = f"https://github.com/{match.group(1)}/{match.group(2)}"
    
    # If still no repo info and we have an env_repo_url, use that
    if (not repo_info["repo_owner"] or not repo_info["repo_name"]) and env_repo_url:
        if env_repo_url.startswith("https://github.com/"):
            match = re.search(r'github\.com/([^/]+)/([^/]+)', env_repo_url)
            if match:
                repo_info["repo_owner"] = match.group(1)
                repo_info["repo_name"] = match.group(2)
                repo_info["github_repo_url"] = env_repo_url
        else:
            # Assume format is owner/repo
            parts = env_repo_url.split('/')
            if len(parts) == 2:
                repo_info["repo_owner"] = parts[0]
                repo_info["repo_name"] = parts[1]
                repo_info["github_repo_url"] = f"https://github.com/{parts[0]}/{parts[1]}"
    
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
    
    # Final fallback: return empty valid structure
    logger.warning(f"Could not parse planner response, using fallback: {clean_response}")
    return {"github": {}, "notion": {}}

def validate_execution_plan(plan: Dict) -> bool:
    """Validate that the execution plan matches the required structure for MCP"""
    if not isinstance(plan, dict):
        return False
    
    # Check for either github or notion section
    has_github = "github" in plan and isinstance(plan["github"], dict)
    has_notion = "notion" in plan and isinstance(plan["notion"], dict)
    
    if not (has_github or has_notion):
        return False
    
    # Validate sections if present
    for section in ["github", "notion"]:
        if section in plan:
            section_plan = plan[section]
            if not isinstance(section_plan, dict):
                return False
            
            for tool_name, tasks in section_plan.items():
                if not isinstance(tasks, dict):
                    return False
                
                for task_key, task_desc in tasks.items():
                    if not isinstance(task_desc, str):
                        return False
    
    return True

def mcp_planner_agent(state: AgentState, llm: ChatGoogleGenerativeAI, env_repo_url: Optional[str] = None) -> Dict[str, Any]:
    """MCP-enabled planner agent that routes queries to appropriate MCP servers"""
    logger.info("Starting MCP planner agent")
    
    user_query = state["user_query"].lower()
    
    # Determine if this is a GitHub, Notion, or mixed query
    github_keywords = ["github", "issue", "pr", "pull request", "repository", "repo", "commit", "branch", "file"]
    notion_keywords = ["notion", "database", "page", "workspace", "properties", "blocks"]
    
    is_github_query = any(keyword in user_query for keyword in github_keywords)
    is_notion_query = any(keyword in user_query for keyword in notion_keywords)
    
    # Extract GitHub repository information if needed
    github_info = {}
    if is_github_query:
        github_info = extract_github_info(state, env_repo_url)
        
        # Check if we have valid repository info for GitHub queries
        if not github_info.get("repo_owner") or not github_info.get("repo_name"):
            logger.warning("No valid GitHub repository info found for GitHub query")
            return {
                **state,
                "execution_plan": {"github": {}, "notion": {}},
                "task_queue": [],
                "errors": ["No valid GitHub repository information found for GitHub query"]
            }
    
    # Check for authentication errors
    if state.get("errors"):
        for error in state["errors"]:
            if "401" in str(error) or "unauthorized" in str(error).lower():
                logger.error("API authentication error detected in previous steps")
                return {
                    **state,
                    "execution_plan": {"github": {}, "notion": {}},
                    "task_queue": [],
                    "errors": state["errors"] + ["API authentication failed. Please check your tokens."]
                }
    
    # Create augmented query with context
    if is_github_query and github_info:
        owner = github_info["repo_owner"]
        repo = github_info["repo_name"]
        augmented_query = f"{state['user_query']} (repository: {owner}/{repo})"
    else:
        augmented_query = state['user_query']
    
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an MCP planner agent for GitHub and Notion operations. You MUST follow these instructions EXACTLY:
        
        ## STRICT OUTPUT REQUIREMENTS
        - OUTPUT ONLY A VALID JSON OBJECT - NO ADDITIONAL TEXT
        - THE JSON MUST HAVE SERVICE KEYS CONTAINING A DICTIONARY OF MCP TOOLS
        - SUPPORTED SERVICE KEYS: "github", "notion"
        - EACH SERVICE DICTIONARY CONTAINS MCP TOOL NAMES AS KEYS
        - YOU CAN USE MULTIPLE TOOLS PER SERVICE PER QUERY
        - USE ONLY THESE EXACT MCP TOOL NAMES (CASE-SENSITIVE):
        
        GITHUB MCP TOOLS:
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
            
        NOTION MCP TOOLS:
            * list_notion_databases
            * get_notion_database
            * query_notion_database
            * create_notion_page
            * get_notion_page
            * update_notion_page
            * search_notion
        
        ## MCP ROUTING RULES
        - Each tool will be executed via MCP server calls
        - Parameters will be extracted from task descriptions
        - Task descriptions must be NATURAL LANGUAGE descriptions
        - DO NOT include parameter values in the task description
        - Each task must have a unique task key (e.g., "mcp_task_1", "mcp_task_2")
        
        ## CORRECT EXAMPLES FOR GITHUB MCP (COPY THIS FORMAT EXACTLY):
        {{"github": {{"get_issue": {{"mcp_task_1": "Get issue #3275 from microsoft/vscode"}}}}}}
        {{"github": {{"search_issues_and_prs": {{"mcp_task_1": "is:pr is:closed repo:QwenLM/Qwen"}}}}}}
        
        ## CORRECT EXAMPLES FOR NOTION MCP (COPY THIS FORMAT EXACTLY):
        {{"notion": {{"list_notion_databases": {{"mcp_task_1": "Find databases related to interview"}}}}}}
        {{"notion": {{"query_notion_database": {{"mcp_task_1": "Query the Tasks database for open items"}}}}}}
        
        ## MIXED SERVICE EXAMPLE:
        {{"github": {{"get_issue": {{"mcp_task_1": "Get issue #123"}}}}, "notion": {{"create_notion_page": {{"mcp_task_2": "Create page with issue details"}}}}}}
        
        If you cannot create a valid plan, return: {{"github": {{}}, "notion": {{}}}}"""),
        ("human", "User Query: {query}")
    ])
    
    try:
        response = llm.invoke(planner_prompt.format_messages(query=augmented_query))
        response_content = response.content
        
        logger.info(f"Raw planner response: {response_content}")
        
        execution_plan = parse_planner_response(response_content)
        logger.info(f"Parsed execution plan: {execution_plan}")
        
        if not validate_execution_plan(execution_plan):
            logger.warning("Invalid execution plan structure, using fallback")
            execution_plan = {"github": {}, "notion": {}}
        
        # Create task queue with MCP routing information
        task_queue = create_mcp_task_queue(execution_plan)
        
        return {
            **state,
            "execution_plan": execution_plan,
            "task_queue": task_queue,
            "current_task_index": 0,
            "repo_owner": github_info.get("repo_owner"),
            "repo_name": github_info.get("repo_name"),
            "github_repo_url": github_info.get("github_repo_url")
        }
        
    except Exception as e:
        logger.error(f"Planner agent failed: {str(e)}")
        return {
            **state,
            "execution_plan": {"github": {}, "notion": {}},
            "task_queue": [],
            "errors": state.get("errors", []) + [f"Planning failed: {str(e)}"]
        }

def create_mcp_task_queue(execution_plan: Dict) -> List[Dict[str, Any]]:
    """Convert execution plan to a flat task queue with MCP routing"""
    task_queue = []
    task_counter = 0
    
    # Process GitHub MCP tasks
    github_section = execution_plan.get("github", {})
    if isinstance(github_section, dict):
        for tool_name, tasks in github_section.items():
            if not isinstance(tasks, dict):
                logger.warning(f"Invalid tasks structure for GitHub tool {tool_name}: {tasks}")
                continue
            
            for task_key, task_description in tasks.items():
                if not isinstance(task_description, str):
                    logger.warning(f"Invalid task description for {tool_name}:{task_key}: {task_description}")
                    continue
                
                task_counter += 1
                task_queue.append({
                    "task_id": f"github_mcp_{task_counter}",
                    "service": "github",
                    "mcp_service": "github",
                    "tool_name": tool_name,
                    "task_key": task_key,
                    "description": task_description,
                    "is_mcp": True
                })
    
    # Process Notion MCP tasks
    notion_section = execution_plan.get("notion", {})
    if isinstance(notion_section, dict):
        for tool_name, tasks in notion_section.items():
            if not isinstance(tasks, dict):
                logger.warning(f"Invalid tasks structure for Notion tool {tool_name}: {tasks}")
                continue
            
            for task_key, task_description in tasks.items():
                if not isinstance(task_description, str):
                    logger.warning(f"Invalid task description for {tool_name}:{task_key}: {task_description}")
                    continue
                
                task_counter += 1
                task_queue.append({
                    "task_id": f"notion_mcp_{task_counter}",
                    "service": "notion", 
                    "mcp_service": "notion",
                    "tool_name": tool_name,
                    "task_key": task_key,
                    "description": task_description,
                    "is_mcp": True
                })
    
    # Log what tasks we're creating
    if task_queue:
        logger.info(f"Created {len(task_queue)} MCP tasks:")
        for task in task_queue:
            logger.info(f"  - {task['task_id']}: {task['mcp_service']}.{task['tool_name']} - {task['description']}")
    else:
        logger.warning("No tasks created from execution plan")
    
    return task_queue

# Backward compatibility
planner_agent = mcp_planner_agent
