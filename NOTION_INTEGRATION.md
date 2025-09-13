# Notion Integration for AI Task Copilot

## Overview
This document describes the integration of Notion functionality into the existing AI Task Copilot system, which previously only supported GitHub operations. The integration allows the system to handle both GitHub and Notion related queries in a unified workflow.

## Changes Made

### 1. Created Notion Tools (`app/tools/notion_tools.py`)
- Created a separate file for Notion tools following the same pattern as GitHub tools
- Implemented 7 Notion tools:
  - `list_notion_databases`: List Notion databases with optional search filter
  - `get_notion_database`: Get details of a specific Notion database
  - `query_notion_database`: Query a Notion database with filters and sorts
  - `create_notion_page`: Create a new page in a Notion database
  - `get_notion_page`: Get details of a specific Notion page
  - `update_notion_page`: Update properties of a Notion page
  - `search_notion`: Search across all Notion pages and databases

### 2. Updated Parent Agent (`app/agents/parent_agent.py`)
- Renamed `GitHubAgentWorkflow` to `MultiServiceAgentWorkflow` for clarity
- Modified the workflow to initialize and combine both GitHub and Notion tools
- Updated the tool mapping to include both service types
- Maintained backward compatibility by keeping the old class name

### 3. Enhanced Planner (`app/agents/planner.py`)
- Updated the planner prompt to support both GitHub and Notion tools
- Modified the prompt to include instructions for Notion-specific operations
- Updated the `create_task_queue` function to handle tasks from both services
- Added proper service identification for each task

### 4. Updated Executor (`app/agents/executor.py`)
- Modified the `task_executor` function to handle both GitHub and Notion tools
- Added service-specific parameter handling
- Updated parameter name mapping to include Notion tool mappings
- Ensured proper tool instantiation for both service types

### 5. Enhanced Formatter (`app/agents/formatter.py`)
- Updated the formatter prompt to handle responses from both GitHub and Notion tools
- Modified fallback formatting to categorize results by service
- Added guidelines for handling Notion entities in responses

### 6. Updated Data Models (`app/agents/utils.py`)
- Modified the `AgentState` TypedDict to support multi-service execution plans
- Updated documentation to reflect support for both GitHub and Notion

### 7. Updated Main Application (`app/main.py`)
- Updated import to use the new `MultiServiceAgentWorkflow` class
- Maintained the same API interface

### 8. Documentation Updates
- Updated README to include Notion integration in the feature list

## Usage

The system now supports queries that involve both GitHub and Notion operations. Examples:

1. GitHub only: "Find all open issues in the repository"
2. Notion only: "List all databases in my Notion workspace"
3. Mixed queries: "Create a GitHub issue for the bug found in Notion page XYZ"

## Environment Variables

To use the Notion functionality, you need to set the `NOTION_API_KEY` environment variable in addition to the existing GitHub and Google API variables.

## Testing

The Notion tools have been tested and verified to load correctly. The system maintains full backward compatibility with existing GitHub-only workflows.