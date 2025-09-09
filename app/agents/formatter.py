# formatter.py
import json
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.utils import AgentState, TaskStatus

logger = logging.getLogger(__name__)

# place near top of file or inside this function
def _extract_notion_title_from_db(db: dict) -> str:
    """
    Try multiple common Notion response shapes to extract a human title.
    Returns "Untitled" if nothing sensible found.
    """
    # 1) Top-level "title" list (database object)
    try:
        if isinstance(db.get("title"), list) and db["title"]:
            title_obj = db["title"][0]
            if isinstance(title_obj, dict):
                # plain_text is the most reliable
                if title_obj.get("plain_text"):
                    return title_obj["plain_text"]
                # fallback: text.content
                text = title_obj.get("text")
                if isinstance(text, dict) and text.get("content"):
                    return text["content"]
    except Exception:
        pass

    # 2) properties.title (common in property-based DB responses)
    try:
        props = db.get("properties", {})
        if isinstance(props, dict) and "title" in props:
            t = props["title"]
            if isinstance(t, dict):
                title_list = t.get("title") or []
                if title_list:
                    first = title_list[0]
                    if isinstance(first, dict):
                        if first.get("plain_text"):
                            return first["plain_text"]
                        if isinstance(first.get("text"), dict) and first["text"].get("content"):
                            return first["text"]["content"]
    except Exception:
        pass

    # 3) Generic scan: look for any property that is type==title or type==rich_text
    try:
        for prop_name, prop_value in (db.get("properties") or {}).items():
            if not isinstance(prop_value, dict):
                continue
            ptype = prop_value.get("type")
            key = None
            if ptype == "title":
                key = "title"
            elif ptype == "rich_text":
                key = "rich_text"
            if key and isinstance(prop_value.get(key), list) and prop_value[key]:
                el = prop_value[key][0]
                if isinstance(el, dict):
                    if el.get("plain_text"):
                        return el["plain_text"]
                    if isinstance(el.get("text"), dict) and el["text"].get("content"):
                        return el["text"]["content"]
    except Exception:
        pass

    # 4) Fallbacks for custom wrappers or different casing (Name, name)
    if db.get("Name"):
        return db.get("Name")
    if db.get("name"):
        return db.get("name")

    # 5) Last fallback: attempt to parse the URL or id -> but return "Untitled" for now
    return "Untitled"


def _format_page_properties(page_properties: dict) -> str:
    """Format individual page properties like the API documentation shows"""
    formatted = ""
    
    for prop_name, prop_data in page_properties.items():
        prop_type = prop_data.get("type", "unknown")
        
        if prop_type == "title" and prop_data.get("title"):
            title_text = prop_data["title"][0].get("plain_text", "")
            formatted += f"**{prop_name}:** {title_text}\n"
            
        elif prop_type == "rich_text" and prop_data.get("rich_text"):
            text_content = "".join([block.get("plain_text", "") for block in prop_data["rich_text"]])
            formatted += f"**{prop_name}:** {text_content}\n"
            
        elif prop_type == "select" and prop_data.get("select"):
            select_value = prop_data["select"].get("name", "")
            formatted += f"**{prop_name}:** {select_value}\n"
            
        elif prop_type == "number" and prop_data.get("number") is not None:
            formatted += f"**{prop_name}:** {prop_data['number']}\n"
            
        elif prop_type == "date" and prop_data.get("date"):
            date_value = prop_data["date"].get("start", "")
            formatted += f"**{prop_name}:** {date_value}\n"
            
        elif prop_type == "checkbox":
            checked = "✓" if prop_data.get("checkbox") else "✗"
            formatted += f"**{prop_name}:** {checked}\n"
            
        elif prop_type == "multi_select" and prop_data.get("multi_select"):
            options = [opt.get("name", "") for opt in prop_data["multi_select"]]
            formatted += f"**{prop_name}:** {', '.join(options)}\n"
            
        elif prop_type == "relation" and prop_data.get("relation"):
            rel_count = len(prop_data["relation"])
            formatted += f"**{prop_name}:** {rel_count} related items\n"
    
    return formatted


def response_formatter(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """Enhanced response formatter with better output categorization"""
    logger.info("Formatting response")
    task_results = state["task_results"]
    errors = state["errors"]
    user_query = state["user_query"]
    execution_plan = state["execution_plan"]
    execution_status = state["execution_status"]
    
    # >>>>> CORRECTED NOTION ERROR HANDLING <<<<<
    # Check for Notion errors by looking for task IDs starting with "notion_"
    for task_id, result in task_results.items():
        if task_id.startswith("notion_"):
            # Handle both dictionary results and string errors
            if isinstance(result, dict) and result.get("error"):
                error_msg = result["error"]
                return {
                    **state,
                    "final_output": f"I'm sorry, I couldn't retrieve the information from your Notion workspace.\n\n{error_msg}",
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=f"I'm sorry, I couldn't retrieve the information from your Notion workspace.\n\n{error_msg}")]
                }
            # Handle cases where the entire result is an error string
            elif isinstance(result, str) and "error" in result.lower():
                return {
                    **state,
                    "final_output": f"I'm sorry, I couldn't retrieve the information from your Notion workspace.\n\n{result}",
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=f"I'm sorry, I couldn't retrieve the information from your Notion workspace.\n\n{result}")]
                }
    
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
    
    # >>>>> NEW: ENHANCED DATABASE DETAILS HANDLING <<<<<
    # Look for get_notion_database results - this handles the "Get details" queries
    for task_id, result in task_results.items():
        if "get_notion_database" in task_id and isinstance(result, dict):
            if "error" in result:
                return {
                    **state,
                    "final_output": f"I couldn't retrieve the database details: {result['error']}",
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=f"I couldn't retrieve the database details: {result['error']}")]
                }
            elif "database" in result:
                database = result["database"]
                
                # Extract database information
                db_title = _extract_notion_title_from_db(database)
                db_id = database.get("id", "Unknown ID")
                db_url = database.get("url", f"https://www.notion.so/{db_id.replace('-', '')}")
                created_time = database.get("created_time", "Unknown")
                last_edited_time = database.get("last_edited_time", "Unknown")
                
                # Format the detailed response
                formatted_response = f"# Database Details: {db_title}\n\n"
                formatted_response += f"**Database ID:** {db_id}\n"
                formatted_response += f"**URL:** {db_url}\n"
                formatted_response += f"**Created:** {created_time}\n"
                formatted_response += f"**Last Modified:** {last_edited_time}\n\n"
                
                # Add properties information
                properties = database.get("properties", {})
                if properties:
                    formatted_response += _format_page_properties(properties)
                else:
                    formatted_response += "**Properties:** No properties defined\n"
                
                # Add description if available
                description = database.get("description", [])
                if description:
                    desc_text = ""
                    for desc_block in description:
                        if isinstance(desc_block, dict) and desc_block.get("plain_text"):
                            desc_text += desc_block["plain_text"]
                    if desc_text:
                        formatted_response += f"\n**Description:** {desc_text}\n"
                
                return {
                    **state,
                    "final_output": formatted_response,
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=formatted_response)]
                }
    
    # >>>>> CORRECTED NOTION DATABASE HANDLING <<<<<
    # Look for query_notion_database results by checking task IDs
    for task_id, result in task_results.items():
        if "query_notion_database" in task_id and isinstance(result, dict) and "results" in result:
            results = result["results"]
            if results:
                formatted_results = f"Found {len(results)} entries in the database:\n\n"
                
                for i, page in enumerate(results[:3], 1):  # Show first 3 entries
                    # Get page title
                    title = "Untitled"
                    if "properties" in page:
                        for prop_name, prop_data in page["properties"].items():
                            if prop_data.get("type") == "title":
                                title = prop_data["title"][0].get("plain_text", "Untitled") if prop_data.get("title") else "Untitled"
                                break
                    
                    formatted_results += f"## {i}. {title}\n"
                    formatted_results += f"**Page ID:** {page.get('id', '')}\n"
                    formatted_results += f"**URL:** {page.get('url', '')}\n"
                    formatted_results += f"**Created:** {page.get('created_time', '')}\n"
                    formatted_results += f"**Last Modified:** {page.get('last_edited_time', '')}\n\n"
                    
                    # Add properties
                    if "properties" in page:
                        formatted_results += _format_page_properties(page["properties"])
                    
                    formatted_results += "\n---\n\n"
                
                if len(results) > 3:
                    formatted_results += f"... and {len(results) - 3} more entries"
                return {
                    **state,
                    "final_output": formatted_results,
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=formatted_results)]
                }
    
    # Special handling for list_notion_databases - FIXED TO SHOW ACTUAL DATABASE NAMES
    for task_id, result in task_results.items():
        if "list_notion_databases" in task_id and isinstance(result, dict):
            if "error" in result:
                return {
                    **state,
                    "final_output": f"I'm sorry, I couldn't retrieve your Notion databases: {result['error']}",
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=f"I'm sorry, I couldn't retrieve your Notion databases: {result['error']}")]
                }
            elif "databases" in result and result["databases"]:
                databases = result["databases"]
                formatted_results = f"Found {len(databases)} database{'s' if len(databases) != 1 else ''} in your Notion workspace:\n\n"
                
                for i, db in enumerate(databases[:10], 1):  # Show up to 10 databases
                    # CORRECTED: Properly extract database title from Notion API response
                    title = _extract_notion_title_from_db(db)

                    # Get the URL from the API response (it's already provided)
                    url = db.get("url", f"https://www.notion.so/{db['id'].replace('-', '')}")
                    
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   ID: {db['id']}\n"
                    formatted_results += f"   URL: {url}\n\n"
                
                if len(databases) > 10:
                    formatted_results += f"... and {len(databases) - 10} more databases"
                
                return {
                    **state,
                    "final_output": formatted_results,
                    "is_complete": True,
                    "messages": state.get("messages", []) + [AIMessage(content=formatted_results)]
                }
    
    try:
        # Determine if this was an informational query or action query
        action_keywords = ["create", "add", "update", "delete", "comment", "post"]
        is_action_query = any(keyword in user_query.lower() for keyword in action_keywords)
        
        formatter_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a response formatter for GitHub and Notion operations.
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
            - For Notion entities, include relevant information about databases or pages and most Importtant-the Title of the database/page(it should not be "untitled" if the database/page has a title)
            - If no relevant results were found, inform the user politely
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
            github_tasks = [task for task in state["task_queue"] if task.get("service", "github") == "github"]
            notion_tasks = [task for task in state["task_queue"] if task.get("service") == "notion"]
            
            if github_tasks:
                final_output += "\nGitHub Results:\n"
                for task_id, result in task_results.items():
                    if task_id.startswith("github_") and result:
                        final_output += f"• {task_id}: {str(result)[:200]}...\n"
            
            if notion_tasks:
                final_output += "\nNotion Results:\n"
                for task_id, result in task_results.items():
                    if task_id.startswith("notion_") and result:
                        # Special handling for Notion results
                        if isinstance(result, dict):
                            if "error" in result:
                                final_output += f"• {task_id}: ERROR - {result['error']}\n"
                            elif "databases" in result:
                                db_count = len(result["databases"])
                                final_output += f"• {task_id}: Found {db_count} databases\n"
                            elif "results" in result:
                                item_count = len(result["results"])
                                final_output += f"• {task_id}: Found {item_count} items\n"
                        else:
                            final_output += f"• {task_id}: {str(result)[:200]}...\n"

      
        if errors:
            # Notion error handling
            if "database_id" in str(errors).lower() and "required" in str(errors).lower():
                return {
                    **state,
                    "final_output": "I need to find the database ID first. I'll search for the database by name and then query it.",
                    "is_complete": False,
                    "messages": state.get("messages", []) + [AIMessage(content="I need to find the database ID first. I'll search for the database by name and then query it.")]
                }
            final_output += f"\nNote: {len(errors)} errors encountered during execution."
    
    return {
        **state,
        "final_output": final_output,
        "messages": state.get("messages", []) + [AIMessage(content=final_output)]
    }