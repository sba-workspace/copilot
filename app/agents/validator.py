#validator.py
import logging
from typing import Dict, Any

from app.agents.utils import AgentState, TaskStatus

logger = logging.getLogger(__name__)

def execution_checker(state: AgentState) -> Dict[str, Any]:
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

def should_continue_execution(state: AgentState) -> str:
    """Decide whether to continue execution or move to validation"""
    if state["is_complete"]:
        return "validate"
    return "continue"

def output_validator(state: AgentState) -> Dict[str, Any]:
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

def should_retry_execution(state: AgentState) -> str:
    """Decide whether to retry execution or format response"""
    if state["is_complete"]:
        return "format"
    else:
        # For retry, reset the task index to restart execution
        state["current_task_index"] = 0
        return "retry"