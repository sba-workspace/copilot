# github_subgraph.py
from typing_extensions import TypedDict, Annotated
from typing import Dict, Any, Optional, List, Type
import re
from langgraph.graph import StateGraph
from langchain.tools import BaseTool
from langchain_community.utilities.github import GitHubAPIWrapper
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
load_dotenv()
# Import your GitHub tools from the implementation we created
from app.tools.github_tools import get_github_tools


# ------------------------
# Helper Functions
# ------------------------
def parse_repo_url(url: str) -> str:
    """
    Accepts:
      - https://github.com/owner/repo
      - git@github.com:owner/repo.git
      - owner/repo
    Returns: owner/repo (no trailing .git)
    """
    url = url.strip()
    # owner/repo already
    if re.match(r"^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$", url):
        return url
    m = re.search(r"github\.com[:/]+([^/]+/[^/]+)(?:\.git)?", url)
    if m:
        return m.group(1).rstrip(".git")
    # fallback raise
    raise ValueError(f"Could not parse repository from: {url}")


# ------------------------
# State Definitions
# ------------------------
class GitHubTaskState(TypedDict):
    task: Dict[str, Any]  # Planner-provided task spec
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    tool_name: str  # Store tool name, not the tool instance


class SharedState(TypedDict):
    github_results: Dict[str, Any]  # Keyed by task ID


# ------------------------
# Helper Functions
# ------------------------
def parse_repo_from_task(task: Dict[str, Any]) -> str:
    """Extract repository from task parameters"""
    repo_url = task.get("repo_url") or task.get("repository")
    if not repo_url:
        raise ValueError("Task missing repository information")
    return parse_repo_url(repo_url)


def parse_tool_args(tool: BaseTool, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map task parameters to tool-specific inputs.

    Handles args_schema being either a Pydantic BaseModel type or instance, and
    supports Pydantic v1/v2 attribute names.
    """
    try:
        schema_fields = set()

        schema = getattr(tool, "args_schema", None)

        # If args_schema is a Pydantic model class (common pattern)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic v2: model_fields; v1: __fields__
            if hasattr(schema, "model_fields"):
                schema_fields = set(schema.model_fields.keys())
            elif hasattr(schema, "__fields__"):
                schema_fields = set(schema.__fields__.keys())

        # If args_schema is an instance of a BaseModel (less common)
        elif isinstance(schema, BaseModel):
            if hasattr(schema, "model_fields"):
                schema_fields = set(schema.model_fields.keys())
            elif hasattr(schema, "__fields__"):
                schema_fields = set(schema.__fields__.keys())

        # If no schema info, accept no args (empty)
        if not schema_fields:
            return {}

        return {k: v for k, v in task.items() if k in schema_fields}
    except Exception as e:
        raise ValueError(f"Invalid parameters for {getattr(tool, 'name', str(tool))}: {str(e)}")


# ------------------------
# Subgraph Nodes
# ------------------------
def validate_task(state: GitHubTaskState) -> GitHubTaskState:
    """Ensure task has required fields"""
    task = state["task"]
    if not all(k in task for k in ["service", "tool", "task"]):
        state["error"] = "Invalid task format - missing service/tool/task"
        return state

    if task["service"].lower() != "github":
        state["error"] = "Invalid service - only GitHub supported"
        return state

    # Store tool name, don't keep tool instance in state
    state["tool_name"] = task["tool"]

    # Validate tool exists
    available_tools = {tool.name: tool for tool in get_github_tools()}
    if task["tool"] not in available_tools:
        state["error"] = f"Unknown GitHub tool: {task['tool']}"

    return state


def configure_github(state: GitHubTaskState) -> GitHubTaskState:
    """Set up GitHub environment for the task's repository"""
    if not state.get("error"):
        try:
            # Validate repo string early
            _ = parse_repo_from_task(state["task"])
            # We'll handle repo configuration during tool execution
        except Exception as e:
            state["error"] = f"Repository configuration failed: {str(e)}"
    return state


def execute_tool(state: GitHubTaskState) -> GitHubTaskState:
    """Run the selected GitHub tool with task parameters"""
    if not state.get("error"):
        try:
            # Get fresh tool instances (safe for concurrency)
            available_tools = {tool.name: tool for tool in get_github_tools()}
            tool = available_tools[state["tool_name"]]

            # Configure repo for this specific execution
            repo = parse_repo_from_task(state["task"])
            wrapper = GitHubAPIWrapper(github_repository=repo)

            # Inject wrapper/github client into tool if expected
            try:
                if hasattr(tool, "_wrapper"):
                    setattr(tool, "_wrapper", wrapper)
                if hasattr(tool, "github"):
                    setattr(tool, "github", wrapper.github)
            except Exception:
                pass

            # Parse and validate arguments (only include expected fields)
            args = parse_tool_args(tool, state["task"])

            # --- Robust tool invocation (handles different BaseTool.run signatures) ---
            schema = getattr(tool, "args_schema", None)

            # If tool expects a Pydantic model instance (common), build it and pass as single arg
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                try:
                    model_input = schema(**args) if args else schema()
                except Exception as e:
                    raise ValidationError(f"Failed to construct args model for {tool.name}: {e}")

                # Try calling with the model instance first
                try:
                    state["result"] = tool.run(model_input)
                except TypeError:
                    # fallback: maybe the tool expects kwargs
                    state["result"] = tool.run(**(args or {}))
            else:
                # No schema or unknown schema type: try kwargs, then try single None input
                try:
                    state["result"] = tool.run(**(args or {}))
                except TypeError:
                    try:
                        state["result"] = tool.run(None)
                    except TypeError as te:
                        # give up with a clear error
                        raise RuntimeError(f"Tool.run signature not supported for tool {tool.name}: {te}")

        except ValidationError as ve:
            state["error"] = f"Validation error: {ve}"
        except Exception as e:
            state["error"] = f"Tool execution failed: {str(e)}"
    return state


def finalize_result(state: GitHubTaskState) -> GitHubTaskState:
    """Format final output for integration with parent graph"""
    if state.get("error"):
        state["result"] = {"success": False, "error": state["error"], "tool": state.get("tool_name")}
    else:
        # Ensure result is JSON-serializable (our tools already do this)
        state["result"] = {"success": True, "data": state["result"], "tool": state.get("tool_name")}
    return state


# ------------------------
# Subgraph Construction
# ------------------------
def build_github_subgraph() -> StateGraph:
    builder = StateGraph(GitHubTaskState)

    # Node 1: Validate task structure
    builder.add_node("validate_task", validate_task)

    # Node 2: Configure GitHub connection
    builder.add_node("configure_github", configure_github)

    # Node 3: Execute the tool
    builder.add_node("execute_tool", execute_tool)

    # Node 4: Final formatting
    builder.add_node("finalize_result", finalize_result)

    # Edges
    builder.add_edge("validate_task", "configure_github")
    builder.add_edge("configure_github", "execute_tool")
    builder.add_edge("execute_tool", "finalize_result")

    builder.set_entry_point("validate_task")
    builder.set_finish_point("finalize_result")

    return builder.compile()


# ------------------------
# Parent Graph Integration Example
# ------------------------
def build_parent_graph() -> StateGraph:
    class ParentState(TypedDict):
        tasks: Annotated[List[Dict[str, Any]], "List of tasks to process"]
        results: Annotated[Dict[str, Any], "Results keyed by task ID"]
        current_task_index: Annotated[int, "Index of the current task being processed"]

    builder = StateGraph(ParentState)

    # Add the GitHub subgraph as a node
    github_subgraph = build_github_subgraph()

    def execute_github_task(state: ParentState) -> Dict:
        """Wrapper to connect parent state to subgraph"""
        if state["current_task_index"] >= len(state["tasks"]):
            return {"error": "No more tasks to process"}

        task = state["tasks"][state["current_task_index"]]
        subgraph_state = {
            "task": task,
            "result": None,
            "error": None,
            "tool_name": ""
        }

        # Run the subgraph and capture finished state
        subgraph_result = github_subgraph.invoke(subgraph_state)

        # Merge result into parent's results
        updated_results = {
            **state.get("results", {}),
            f"task_{state['current_task_index']}": subgraph_result.get("result")
        }

        return {
            "tasks": state["tasks"],  # preserve tasks
            "results": updated_results,
            "current_task_index": state["current_task_index"] + 1
        }

    builder.add_node("github_executor", execute_github_task)

    # Check if we have more tasks
    def has_more_tasks(state: ParentState) -> str:
        return "github_executor" if state["current_task_index"] < len(state["tasks"]) else "finish"

    # Trivial start node
    def start_node(state: ParentState) -> Dict:
        return {"tasks": state.get("tasks", []), "results": state.get("results", {}), "current_task_index": 0}

    builder.add_node("start", start_node)
    builder.add_node("finish", lambda x: x)

    # Edges
    builder.set_entry_point("start")
    builder.add_edge("start", "github_executor")
    builder.add_conditional_edges(
        "github_executor",
        has_more_tasks,
        {
            "github_executor": "github_executor",
            "finish": "finish"
        }
    )
    builder.set_finish_point("finish")

    return builder.compile()


# ------------------------
# Usage Example
# ------------------------
if __name__ == "__main__":
    # Sample task from planner
    sample_tasks = [
        {
            "service": "GitHub",
            "tool": "get_issues",
            "task": "Get open issues",
            "repo_url": "https://github.com/langchain-ai/langchain"
        },
        {
            "service": "GitHub",
            "tool": "search_issues_and_prs",
            "task": "Search for documentation issues",
            "query": "label:documentation",
            "repo_url": "https://github.com/langchain-ai/langchain"
        }
    ]

    # Initialize parent graph
    parent_graph = build_parent_graph()

    # Run the workflow
    initial_state = {"tasks": sample_tasks}
    result = parent_graph.invoke(initial_state)

    print("Final Results:")
    for task_id, res in result["results"].items():
        print(f"\n{task_id}:")
        if not res:
            print("  (no result)")
            continue
        if isinstance(res, dict) and res.get("success") is False:
            print(f"  ERROR: {res.get('error')}")
        else:
            print(f"  SUCCESS: {res.get('data')}")
