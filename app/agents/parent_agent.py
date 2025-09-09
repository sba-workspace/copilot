# parent_agent.py

import os
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

# Import our modularized components
from app.agents.utils import AgentState
from app.agents.planner import planner_agent
from app.agents.executor import task_executor
from app.agents.validator import execution_checker, should_continue_execution, output_validator, should_retry_execution
from app.agents.formatter import response_formatter
from app.tools.github_tools import get_github_tools
from app.tools.notion_tools import get_notion_tools
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
os.environ["LANGSMITH_TRACING"] = "true"

logger = logging.getLogger(__name__)

class MultiServiceAgentWorkflow:
    def __init__(self, llm_model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        # Initialize both GitHub and Notion tools
        self.github_tools = get_github_tools()
        self.notion_tools = get_notion_tools()
        
        # Combine tools from both services
        all_tools = self.github_tools + self.notion_tools
        self.tool_map = {tool.name.lower(): tool for tool in all_tools}
        self.env_repo_url = os.getenv("GITHUB_REPO_URL")
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the main workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner_agent", lambda state: planner_agent(
            state, 
            self.llm,
            self.env_repo_url
        ))
        workflow.add_node("task_executor", lambda state: task_executor(
            state, 
            self.tool_map,
            self.env_repo_url
        ))
        workflow.add_node("execution_checker", execution_checker)
        workflow.add_node("output_validator", output_validator)
        workflow.add_node("response_formatter", lambda state: response_formatter(
            state, 
            self.llm
        ))
        
        # Add edges
        workflow.add_edge(START, "planner_agent")
        workflow.add_edge("planner_agent", "task_executor")
        workflow.add_edge("task_executor", "execution_checker")
        workflow.add_conditional_edges(
            "execution_checker",
            should_continue_execution,
            {
                "continue": "task_executor",
                "validate": "output_validator"
            }
        )
        workflow.add_conditional_edges(
            "output_validator",
            should_retry_execution,
            {
                "retry": "task_executor", 
                "format": "response_formatter"
            }
        )
        workflow.add_edge("response_formatter", END)
        
        return workflow.compile()
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """Run the complete workflow supporting both GitHub and Notion"""
        logger.info(f"Starting multi-service workflow for query: {user_query}")
        env_repo_id = os.getenv("GITHUB_REPO_URL")
        
        # Check if Notion API key is set
        notion_api_key = os.getenv("NOTION_API_KEY")
        has_notion = notion_api_key is not None and notion_api_key.strip() != ""
        
        # Check if GitHub repo is configured
        has_github = bool(os.getenv("GITHUB_REPO_URL"))
        
        # If the query is clearly about GitHub but GitHub isn't configured
        if not has_github and any(word in user_query.lower() for word in ["github", "issue", "pr", "pull request", "repository", "repo"]):
            return {
                "success": False,
                "response": "GitHub integration is not configured. Please set the GITHUB_REPO_URL environment variable to use GitHub functionality.",
                "services_available": {"github": False, "notion": has_notion},
                "execution_plan": {},
                "task_results": {},
                "execution_status": {},
                "errors": ["GITHUB_REPO_URL not configured"],
                "messages": [{"type": "human", "content": user_query}],
                "repo_info": {}
            }
        
        # Initialize state with environment info
        initial_state = {
            "services_available": {
                "github": has_github,
                "notion": has_notion
            },
            "user_query": user_query,
            "github_repo_url": None,
            "execution_plan": {},
            "current_service": None,
            "current_tool": None,
            "current_task_key": None,
            "task_queue": [],
            "current_task_index": 0,
            "repo_owner": None,
            "repo_name": None,
            "repo_instance": None,
            "task_results": {},
            "shared_context": {},
            "execution_status": {},
            "is_complete": False,
            "validation_attempts": 0,
            "max_validation_attempts": 3,
            "final_output": None,
            "messages": [{"type": "human", "content": user_query}],
            "errors": []
        }
        
        # Set GitHub repo info if available
        if env_repo_id:
            # Parse the repo identifier (format: owner/repo)
            repo_parts = env_repo_id.split('/')
            if len(repo_parts) == 2:
                repo_owner = repo_parts[0]
                repo_name = repo_parts[1]
                # Set GITHUB_REPOSITORY for GitHub tools (they expect this format)
                repo_full_name = f"{repo_owner}/{repo_name}"
                os.environ["GITHUB_REPOSITORY"] = repo_full_name
                
                initial_state.update({
                    "github_repo_url": f"https://github.com/{repo_full_name}",
                    "repo_owner": repo_owner,
                    "repo_name": repo_name,
                    "shared_context": {
                        "repo_owner": repo_owner,
                        "repo_name": repo_name,
                        "repo_url": f"https://github.com/{repo_full_name}"
                    }
                })
                # Create GitHub tools with the specific repository
                self.github_tools = get_github_tools(repo=repo_full_name)
                # Rebuild tool map with updated GitHub tools
                all_tools = self.github_tools + self.notion_tools
                self.tool_map = {tool.name.lower(): tool for tool in all_tools}
        
        try:
            final_state = self.workflow.invoke(initial_state)
            return {
                "success": True,
                "response": final_state.get("final_output", "Task completed"),
                "execution_plan": final_state.get("execution_plan", {}),
                "task_results": final_state.get("task_results", {}),
                "execution_status": final_state.get("execution_status", {}),
                "errors": final_state.get("errors", []),
                "messages": final_state.get("messages", []),
                "repo_info": {
                    "owner": final_state.get("repo_owner"),
                    "name": final_state.get("repo_name"),
                    "url": final_state.get("github_repo_url")
                }
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "response": f"Workflow failed: {str(e)}",
                "execution_plan": {},
                "task_results": {},
                "execution_status": {},
                "errors": [str(e)],
                "messages": [],
                "repo_info": {}
            }

# Keep the old class name for backward compatibility
GitHubAgentWorkflow = MultiServiceAgentWorkflow