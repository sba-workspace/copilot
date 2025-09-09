from enum import Enum
from typing import Optional, List, Dict, Any, TypedDict,Annotated
from pydantic import BaseModel, Field

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
    execution_plan: Dict[str, Dict[str, Dict[str, str]]]  # {"github": {"tool_name": {"task_key": "task_description"}}, "notion": {...}}
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
    messages: Annotated[List[Dict[str, Any]], "add_messages"]
    # Error handling
    errors: List[str]
    # Notion database context - CRITICAL ADDITIONS
    database_id_map: Optional[List[Dict[str, str]]]  # Changed to list of dicts for proper structure
    database_name_map: Optional[Dict[str, str]]  # Name to ID mapping
    pending_db_resolution: Optional[str]  # For tracking which database we're resolving