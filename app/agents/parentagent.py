# app/sgents/parentagent.py
# Enhanced LangGraph workflow for GitHub integration with multi-agent architecture

import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.output_parsers import PydanticOutputParser

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["LANGSMITH_TRACING"] = "true"
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import json
import logging
import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
# Import your GitHub tools
from app.tools.github_tools import get_github_tools,get_wrapper

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    
class Task(BaseModel):
    task_description: str = Field(description="A detailed description of the task.")

class Tool(BaseModel):
    tasks: Dict[str, str] = Field(description="A dictionary of tasks to be performed by the tool.")

class ExecutionPlan(BaseModel):
    github: Dict[str, Tool] = Field(description="The GitHub service containing tools and tasks.")


class AgentState(TypedDict):
    """Shared state across all agents in the workflow"""
    # Input
    user_query: str
    github_repo_url: Optional[str]
    
    # Planning - Updated structure to match your requirements
    execution_plan: Dict[str, Dict[str, Dict[str, str]]]  # {"github": {"tool_name": {"task_key": "task_description"}}}
    current_service: Optional[str]
    current_tool: Optional[str]
    current_task_key: Optional[str]
    task_queue: List[Dict[str, Any]]  # Flattened task queue for execution
    current_task_index: int
    
    # Repository context
    repo_owner: Optional[str]
    repo_name: Optional[str]
    repo_instance: Optional[Any]
    
    # Execution
    task_results: Dict[str, Any]  # Store results from each task
    shared_context: Dict[str, Any]  # Context shared between tasks
    execution_status: Dict[str, str]  # Track status of each task
    
    # Validation & Output
    is_complete: bool
    validation_attempts: int
    max_validation_attempts: int
    final_output: Optional[str]
    
    # Messages for agent communication
    messages: List[BaseMessage]
    
    # Error handling
    errors: List[str]

class GitHubAgentWorkflow:
    def __init__(self, llm_model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.output_parser = PydanticOutputParser(pydantic_object=ExecutionPlan)
        self.github_tools = get_github_tools()
        self.tool_executor = ToolNode(self.github_tools)
        self.workflow = self._create_workflow()
        
        # Create tool mapping for easier access
        self.tool_map = {tool.name.lower(): tool for tool in self.github_tools}
        
    def _create_workflow(self) -> StateGraph:
        """Create the main workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner_agent", self._planner_agent)
        workflow.add_node("task_executor", self._task_executor)
        workflow.add_node("execution_checker", self._execution_checker)
        workflow.add_node("output_validator", self._output_validator)
        workflow.add_node("response_formatter", self._response_formatter)
        
        # Add edges
        workflow.add_edge(START, "planner_agent")
        workflow.add_edge("planner_agent", "task_executor")
        workflow.add_edge("task_executor", "execution_checker")
        workflow.add_conditional_edges(
            "execution_checker",
            self._should_continue_execution,
            {
                "continue": "task_executor",
                "validate": "output_validator"
            }
        )
        workflow.add_conditional_edges(
            "output_validator",
            self._should_retry_execution,
            {
                "retry": "task_executor", 
                "format": "response_formatter"
            }
        )
        workflow.add_edge("response_formatter", END)
        
        return workflow.compile()

    def _extract_github_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract GitHub repository information from text"""
        # Pattern to match GitHub URLs
        github_pattern = r'(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/\s]+?)(?:/|$|\s)'
        match = re.search(github_pattern, text)
        
        if match:
            return {
                "owner": match.group(1),
                "repo": match.group(2).rstrip('.git'),
                "url": f"https://github.com/{match.group(1)}/{match.group(2).rstrip('.git')}"
            }
        return {"owner": None, "repo": None, "url": None}

    def _parse_planner_response(self, response_content: str) -> Dict:
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
    
    def _validate_execution_plan(self, plan: Dict) -> bool:
        """Validate that the execution plan matches the required structure"""
        if not isinstance(plan, dict):
            return False
        
        github_section = plan.get("github")
        if not isinstance(github_section, dict):
            return False
        
        for tool_name, tasks in github_section.items():
            if not isinstance(tasks, dict):
                return False
            for task_key, task_desc in tasks.items():
                if not isinstance(task_key, str) or not isinstance(task_desc, str):
                    return False
        
        return True

    def _planner_agent(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced planner agent using the context prompt structure"""
        logger.info("Starting enhanced planner agent")
        
        # Extract GitHub repository information
        github_info = self._extract_github_info(state["user_query"])
        
        # Use the output parser's format instructions in the prompt
        # format_instructions = self.output_parser.get_format_instructions()
        
        if state.get("github_repo_url"):
            github_info = self._extract_github_info(state["github_repo_url"])
        
        planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a planner agent for GitHub operations. You MUST follow these instructions EXACTLY:

        ## STRICT OUTPUT REQUIREMENTS
        - OUTPUT ONLY A VALID JSON OBJECT - NO ADDITIONAL TEXT
        - THE JSON MUST HAVE A TOP-LEVEL "github" KEY CONTAINING A DICTIONARY
        - THE "github" DICTIONARY CONTAINS TOOL NAMES AS KEYS
        - USE ONLY THESE EXACT TOOL NAMES (CASE-SENSITIVE):
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
        - EACH TOOL NAME MAPS TO A DICTIONARY OF TASKS
        - EACH TASK HAS A UNIQUE KEY AND A DESCRIPTION STRING

        ## CORRECT EXAMPLE (COPY THIS FORMAT EXACTLY):
        {{"github": {{"get_issue": {{"task_1": "Get issue #3275 from microsoft/vscode"}}}}}}

        ## INCORRECT EXAMPLES (NEVER DO THIS):
        - {{"github": "get_issue"}}
        - {{"github": ["get_issue"]}}
        - {{"github": {{"get_issue"}}}}
        - {{"github": {{"get_issue": "Get issue #3275"}}}}

        ## VALIDATION RULES
        1. Must start with {{ and end with }}
        2. Must contain "github" as a key with an object value
        3. Each tool must map to a dictionary of tasks
        4. Each task must have a string description

        If you cannot create a valid plan, return: {{"github": {{}}}}

        NEVER output anything except a valid JSON object."""),
        ("human", "User Query: {query}\nRepository: {owner}/{repo}")
    ])
        
        try:
            response = self.llm.invoke(planner_prompt.format_messages(
            query=state["user_query"],
            owner=github_info["owner"] or "Not provided",
            repo=github_info["repo"] or "Not provided"
        ))
        
            logger.debug(f"Raw planner response: {response.content}")
        
            # Parse the execution plan
            plan_text = response.content.strip()
            
            # Handle common malformed responses
            if plan_text == '"github"':
                logger.warning("LLM returned invalid minimal response, using fallback structure")
                execution_plan = {"github": {}}
            elif plan_text.startswith('"') and plan_text.endswith('"'):
                # It's a string, not an object - try to fix
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
            
            #CRITICAL FIX: Ensure proper nested structure
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
            if not isinstance(execution_plan, dict) or "github" not in execution_plan:
                logger.warning("Execution plan missing github key, creating default structure")
                execution_plan = {"github": {}}
            if not isinstance(execution_plan["github"], dict):
                logger.warning("github value is not a dictionary, resetting")
                execution_plan["github"] = {}
            
            # Convert plan to task queue
            task_queue = self._create_task_queue(execution_plan)
            
            return {
                **state,
                "execution_plan": execution_plan,
                "task_queue": task_queue,
                "current_task_index": 0,
                "repo_owner": github_info["owner"],
                "repo_name": github_info["repo"],
                "github_repo_url": github_info["url"],
                "task_results": {},
                "shared_context": {
                    "repo_owner": github_info["owner"],
                    "repo_name": github_info["repo"],
                    "repo_url": github_info["url"]
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
        
    def _create_task_queue(self, execution_plan: Dict) -> List[Dict[str, Any]]:
        """Convert execution plan to a flat task queue with robust error handling"""
        task_queue = []
        
        # Ensure we have a valid github section
        github_section = execution_plan.get("github", {})
        if not isinstance(github_section, dict):
            logger.warning(f"GitHub section is not a dictionary: {type(github_section)}")
            github_section = {}
        
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
        
        # Log what tasks we're creating
        if task_queue:
            logger.info(f"Created {len(task_queue)} tasks from plan")
            for i, task in enumerate(task_queue):
                logger.debug(f"Task {i+1}: {task['tool_name']} - {task['task_key']}: {task['task_description'][:50]}...")
        else:
            logger.warning("No tasks created from execution plan")
        
        return task_queue

    def _extract_parameters_from_query(self, user_query: str, task_description: str) -> Dict[str, Any]:
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
        file_match = re.search(r'(?:file\s*(?:called\s*)?["\']([^"\']+)["\']|([a-zA-Z0-9_.-]+\.[a-zA-Z]+))', user_query)
        if file_match:
            params["path"] = file_match.group(1) or file_match.group(2)
            
        # Extract search queries
        if "search" in task_description.lower():
            search_match = re.search(r'["\']([^"\']+)["\']', user_query)
            if search_match:
                params["query"] = search_match.group(1)
                
        # Extract comments
        if "comment" in task_description.lower():
            comment_match = re.search(r'saying\s*["\']([^"\']+)["\']', user_query)
            if comment_match:
                params["comment"] = comment_match.group(1)
                
        return params

    def _task_executor(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced task executor with better parameter handling"""
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
            
            # CRITICAL: Configure GitHub wrapper with correct repository FIRST
            if "repo_owner" in shared_context and "repo_name" in shared_context:
                # Get a wrapper instance and configure it with the repository
                wrapper = get_wrapper()
                wrapper.github_repository = f"{shared_context['repo_owner']}/{shared_context['repo_name']}"
                logger.info(f"Configured GitHub wrapper for repository: {wrapper.github_repository}")
            
            # Get the tool instance
            if tool_name_normalized not in self.tool_map:
                error_msg = f"Tool {tool_name} not found"
                execution_status[task_id] = TaskStatus.FAILED.value
                return {
                    **state,
                    "execution_status": execution_status,
                    "errors": errors + [error_msg],
                    "current_task_index": current_index + 1
                }
            
            tool = self.tool_map[tool_name_normalized]
            
            # Extract parameters from user query and shared context
            params = self._extract_parameters_from_query(
                state["user_query"], 
                current_task["task_description"]
            )
            
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
            
            # Execute the tool
            try:
                if hasattr(tool.args_schema, 'model_fields') and tool.args_schema.model_fields:
                    # Tool has parameters - only pass relevant ones
                    filtered_params = {}
                    for field_name in tool.args_schema.model_fields.keys():
                        if field_name in params:
                            filtered_params[field_name] = params[field_name]
                    result = tool.run(filtered_params)
                else:
                    # Tool has no parameters
                    result = tool.run({})
                
                # Store result
                task_results[task_id] = result
                execution_status[task_id] = TaskStatus.COMPLETED.value
                
                # Update shared context with relevant information
                self._update_shared_context(shared_context, tool_name_normalized, result)
                
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

    def _update_shared_context(self, shared_context: Dict[str, Any], tool_name: str, result: Any):
        """Update shared context based on tool results"""
        if tool_name == "get_issue" and isinstance(result, dict):
            shared_context["current_issue"] = result
            if "number" in result:
                shared_context["issue_number"] = result["number"]
                
        elif tool_name == "get_pull_request" and isinstance(result, dict):
            shared_context["current_pr"] = result
            if "number" in result:
                shared_context["pr_number"] = result["number"]
                
        elif tool_name == "read_file" and isinstance(result, dict):
            shared_context["file_content"] = result
            
        elif tool_name == "search_issues_and_prs" and isinstance(result, list) and result:
            shared_context["search_results"] = result
            # Get the most recent issue/PR for potential follow-up actions
            if result:
                shared_context["latest_search_result"] = result[0]
                if "number" in result[0]:
                    shared_context["issue_number"] = result[0]["number"]

    def _execution_checker(self, state: AgentState) -> Dict[str, Any]:
        """Check if all tasks in the queue are completed"""
        logger.info("Checking execution status")
        
        task_queue = state["task_queue"]
        current_index = state["current_task_index"]
        
        # Check if we've processed all tasks
        is_complete = current_index >= len(task_queue)
        
        if is_complete:
            # Count successful vs failed tasks for reporting
            execution_status = state["execution_status"]
            completed_count = sum(1 for status in execution_status.values() if status == TaskStatus.COMPLETED.value)
            failed_count = sum(1 for status in execution_status.values() if status == TaskStatus.FAILED.value)
            
            logger.info(f"All tasks processed: {completed_count} completed, {failed_count} failed")
        
        return {
            **state,
            "is_complete": is_complete
        }

    def _should_continue_execution(self, state: AgentState) -> str:
        """Decide whether to continue execution or move to validation"""
        if state["is_complete"]:
            return "validate"
        return "continue"

    def _output_validator(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced output validator"""
        logger.info("Validating output")
        
        validation_attempts = state["validation_attempts"]
        max_attempts = state["max_validation_attempts"]
        
        if validation_attempts >= max_attempts:
            logger.warning("Max validation attempts reached")
            return {
                **state,
                "final_output": "Maximum validation attempts reached. Proceeding with available results."
            }
        
        # Enhanced validation logic
        task_results = state["task_results"]
        errors = state["errors"]
        execution_status = state["execution_status"]
        
        # Count successful results
        successful_results = sum(1 for status in execution_status.values() if status == TaskStatus.COMPLETED.value)
        failed_results = sum(1 for status in execution_status.values() if status == TaskStatus.FAILED.value)
        total_tasks = len(state["task_queue"])
        
        # Validation criteria
        has_some_results = successful_results > 0
        majority_successful = successful_results >= (total_tasks * 0.5)  # At least 50% success
        no_critical_errors = len(errors) < 3  # Allow some minor errors
        
        is_valid = has_some_results and majority_successful and no_critical_errors
        
        logger.info(f"Validation result: {is_valid} (successful: {successful_results}/{total_tasks}, errors: {len(errors)})")
        
        return {
            **state,
            "validation_attempts": validation_attempts + 1,
            "is_complete": is_valid
        }

    def _should_retry_execution(self, state: AgentState) -> str:
        """Decide whether to retry execution or format response"""
        if state["is_complete"]:
            return "format"
        else:
            # For retry, reset the task index to restart execution
            state["current_task_index"] = 0
            return "retry"

    def _response_formatter(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced response formatter with better output categorization"""
        logger.info("Formatting response")
        
        task_results = state["task_results"]
        errors = state["errors"]
        user_query = state["user_query"]
        execution_plan = state["execution_plan"]
        execution_status = state["execution_status"]
        
        try:
            # Determine if this was an informational query or action query
            action_keywords = ["create", "add", "update", "delete", "comment", "post"]
            is_action_query = any(keyword in user_query.lower() for keyword in action_keywords)
            
            formatter_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a response formatter for GitHub operations.
                Based on the user query and execution results, provide a clear, helpful response.
                
                Query Type: {"Action Query" if is_action_query else "Information Query"}
                
                For Information Queries: Provide the specific information requested in a clear, structured format.
                For Action Queries: Confirm what actions were completed successfully and mention any that failed.
                
                Guidelines:
                - If asking about a specific issue/PR/file, provide the relevant details
                - If performing actions, confirm completion status
                - Keep responses conversational and user-friendly
                - Include key information from results but don't overwhelm with technical details
                - If there were errors, mention them briefly but focus on successful results
                - For GitHub entities, include relevant URLs when available
                """),
                ("human", """
                User Query: {query}
                
                Execution Plan: {plan}
                
                Task Results: {results}
                
                Execution Status: {status}
                
                Errors (if any): {errors}
                
                Please format a helpful response for the user.
                """)
            ])
            
            response = self.llm.invoke(formatter_prompt.format_messages(
                query=user_query,
                plan=json.dumps(execution_plan, indent=2),
                results=json.dumps(task_results, indent=2, default=str),
                status=json.dumps(execution_status, indent=2),
                errors=json.dumps(errors, indent=2) if errors else "None"
            ))
            
            final_output = response.content
            
        except Exception as e:
            logger.error(f"Response formatting failed: {str(e)}")
            # Fallback formatting
            successful_tasks = sum(1 for status in execution_status.values() if status == TaskStatus.COMPLETED.value)
            total_tasks = len(state["task_queue"])
            
            if is_action_query:
                final_output = f"Task execution completed. {successful_tasks}/{total_tasks} tasks completed successfully."
            else:
                # Try to extract key information from results
                final_output = "Here are the results:\n\n"
                for task_id, result in task_results.items():
                    if result:
                        final_output += f"• {task_id}: {str(result)[:200]}...\n"
            
            if errors:
                final_output += f"\nNote: {len(errors)} errors encountered during execution."
        
        return {
            **state,
            "final_output": final_output,
            "messages": state.get("messages", []) + [AIMessage(content=final_output)]
        }

    def run(self, user_query: str, github_repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete workflow"""
        logger.info(f"Starting GitHub workflow for query: {user_query}")
        
        initial_state = {
            "user_query": user_query,
            "github_repo_url": github_repo_url,
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
            "messages": [HumanMessage(content=user_query)],
            "errors": []
        }
        
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


# Enhanced usage examples and test functions
def test_enhanced_github_workflow():
    """Test the enhanced GitHub workflow with various query types"""
    workflow = GitHubAgentWorkflow()
    
    test_queries = [
        {
            "query": "What is issue number 3275 on https://github.com/microsoft/vscode?",
            "description": "Single issue lookup with URL"
        },
        # {
        #     "query": "Show me all open issues in the repository",
        #     "repo_url": "https://github.com/microsoft/vscode",
        #     "description": "List all issues"
        # },
        {
            "query": "Find all issues related to 'performance' and show me the first 5",
            "repo_url": "https://github.com/microsoft/vscode",
            "description": "Search and filter issues"
        },
        # {
        #     "query": "Read the contents of README.md file",
        #     "repo_url": "https://github.com/microsoft/vscode",
        #     "description": "File reading"
        # },
        {
            "query": "Show me what files changed in PR #156",
            "repo_url": "https://github.com/microsoft/vscode", 
            "description": "PR file analysis"
        }
    ]
    
    for i, test_case in enumerate(test_queries):
        print(f"\n=== Test Case {i+1}: {test_case['description']} ===")
        print(f"Query: {test_case['query']}")
        
        result = workflow.run(
            test_case["query"], 
            test_case.get("repo_url")
        )
        
        print(f"Success: {result['success']}")
        print(f"Response: {result['response'][:300]}...")
        print(f"Execution Plan: {json.dumps(result['execution_plan'], indent=2)}")
        print(f"Tasks Executed: {len(result['task_results'])}")
        
        if result['errors']:
            print(f"Errors: {result['errors']}")
    
    return workflow

if __name__ == "__main__":
    # Initialize and test the enhanced workflow
    # In your main setup, add debug-level logging
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )
    workflow = test_enhanced_github_workflow()