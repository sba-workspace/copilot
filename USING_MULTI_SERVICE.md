# Using Multi-Service Functionality

This document explains how to use the new multi-service capabilities of the AI Task Copilot system.

## Overview

The system now supports both GitHub and Notion operations within a single workflow. You can:

1. Execute GitHub-only queries (existing functionality)
2. Execute Notion-only queries (new functionality)
3. Execute mixed queries that involve both services (new functionality)

## Environment Setup

To use the multi-service functionality, you need to set the following environment variables:

```bash
# GitHub credentials (required for GitHub operations)
GITHUB_API_TOKEN=your_github_token
GITHUB_REPOSITORY=owner/repo  # Default repository

# Google API key (required for LLM operations)
GOOGLE_API_KEY=your_google_api_key

# Notion API key (required for Notion operations)
NOTION_API_KEY=your_notion_api_key
```

## Example Queries

### GitHub-Only Queries
These work exactly as before:

```
"What is issue number 3275"
"Show me all open issues in the repository"
"Find all issues related to 'performance'"
```

### Notion-Only Queries
New queries for Notion operations:

```
"List all databases in my Notion workspace"
"Find databases related to 'project management'"
"Query the 'Tasks' database for open items assigned to me"
"Create a new page in the 'Tasks' database with title 'Review documentation'"
"Search for pages about 'Q3 planning'"
```

### Mixed Service Queries
Queries that involve both services:

```
"Find all open GitHub issues and create a Notion page for each in my 'Tasks' database"
"Search for Notion pages about 'bug fixes' and create GitHub issues for each"
"List all GitHub pull requests and create a summary page in my Notion 'Development' database"
"Get the details of GitHub issue #123 and create a Notion page with the same information"
```

## How It Works

The system uses a LangGraph-based workflow with the following components:

1. **Planner Agent**: Analyzes the query and creates an execution plan that can include both GitHub and Notion tools
2. **Executor Agent**: Executes tasks from the plan, handling both service types appropriately
3. **Validator Agent**: Checks execution status and validates results
4. **Formatter Agent**: Formats the final response, handling results from both services

## Supported Operations

### GitHub Operations
- Issue management (get, list, search, comment)
- Pull request operations (get, list, create, files)
- File operations (read, create, update, delete, list)
- Repository search

### Notion Operations
- Database operations (list, get, query)
- Page operations (get, create, update)
- Search operations (search across workspace)

## Testing

To verify the functionality:

1. Run the test script: `python app/agents/test_multiservice.py`
2. Check that both GitHub and Notion tools load correctly
3. Verify that the MultiServiceAgentWorkflow initializes successfully

## Extending Functionality

To add new tools:

1. For GitHub tools: Add to `app/tools/github_tools.py`
2. For Notion tools: Add to `app/tools/notion_tools.py`
3. Update the planner prompt in `app/agents/planner.py` to include new tool names
4. Add parameter mappings in `app/agents/executor.py` if needed

The system is designed to be easily extensible to other services in the future.