import json
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.utils import AgentState,TaskStatus

logger = logging.getLogger(__name__)

def response_formatter(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """Enhanced response formatter with better output categorization"""
    logger.info("Formatting response")
    task_results = state["task_results"]
    errors = state["errors"]
    user_query = state["user_query"]
    execution_plan = state["execution_plan"]
    execution_status = state["execution_status"]
    
    # Check if we have PR file results
    pr_files_tool = "list_pull_request_files"
    if pr_files_tool in task_results:
        pr_files_result = task_results[pr_files_tool]
        
        # Handle empty PR results
        if isinstance(pr_files_result, list) and len(pr_files_result) == 1:
            if pr_files_result[0].get("is_empty", False):
                return {
                    **state,
                    "final_output": "The pull request exists but doesn't contain any file changes.",
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content="The pull request exists but doesn't contain any file changes.")]
                }
            if "error" in pr_files_result[0] and "not found" in pr_files_result[0]["error"].lower():
                return {
                    **state,
                    "final_output": pr_files_result[0]["error"],
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=pr_files_result[0]["error"])]
                }
        # Handle other PR file errors
        elif isinstance(pr_files_result, list) and len(pr_files_result) > 0 and "error" in pr_files_result[0]:
            return {
                **state,
                "final_output": f"Could not retrieve PR files: {pr_files_result[0]['error']}",
                "is_complete": True,
                "messages": state.get("messages", []) + [AIMessage(content=f"Could not retrieve PR files: {pr_files_result[0]['error']}")]
            }
            
            
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
        
        response = llm.invoke(formatter_prompt.format_messages(
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
            final_output = "Here are the results:\n"
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