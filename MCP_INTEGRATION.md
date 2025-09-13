# MCP (Model Context Protocol) Integration

This document describes the transition from direct tool calls to MCP (Model Context Protocol) architecture.

## Overview

The project has been updated to use MCP instead of direct GitHub and Notion API calls. This provides better modularity, scalability, and separation of concerns.

## Architecture Changes

### Before (Direct Tools)
```
User Query → Planner → Executor → GitHub/Notion APIs → Response
```

### After (MCP-based)
```
User Query → MCP Planner → MCP Executor → MCP Servers → GitHub/Notion APIs → Response
```

## Components

### 1. MCP Servers
- **GitHub MCP Server** (`app/mcp/github_mcp_server.py`): Handles all GitHub operations
- **Notion MCP Server** (`app/mcp/notion_mcp_server.py`): Handles all Notion operations

### 2. MCP Client
- **MCP Client** (`app/mcp/mcp_client.py`): Communicates with MCP servers

### 3. Updated Agents
- **MCP Planner** (`app/agents/planner_mcp.py`): Routes queries to appropriate MCP servers
- **MCP Executor** (`app/agents/executor_mcp.py`): Executes tasks via MCP calls

### 4. Configuration
- **MCP Config** (`app/mcp/config.py`): Server configuration and environment validation
- **Server Manager** (`app/mcp/server_manager.py`): Manages MCP server lifecycle

## Environment Variables

Make sure these environment variables are set:

```bash
# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_REPOSITORY=owner/repo  # e.g., microsoft/vscode

# Notion Configuration  
NOTION_API_KEY=your_notion_api_key

# MCP Server Ports (optional)
GITHUB_MCP_PORT=3001
NOTION_MCP_PORT=3002
```

## Installation

1. Update dependencies:
```bash
poetry install
```

2. The MCP package will be installed automatically via pyproject.toml.

## Usage

### Starting MCP Servers

Option 1: Use the server manager
```bash
python -m app.mcp.server_manager
```

Option 2: Start servers individually
```bash
# Terminal 1: GitHub MCP Server
python -m app.mcp.github_mcp_server

# Terminal 2: Notion MCP Server  
python -m app.mcp.notion_mcp_server
```

### Using the Application

The main application interface remains the same:

```python
from app.agents.parent_agent import MultiServiceAgentWorkflow

workflow = MultiServiceAgentWorkflow()
result = workflow.run("Get issue #123 from microsoft/vscode")
```

## MCP Tools Available

### GitHub MCP Tools
- `get_issues` - Get open issues from repository
- `get_issue` - Get specific issue by number
- `comment_on_issue` - Add comment to issue
- `search_issues_and_prs` - Search issues and PRs
- `list_open_pull_requests` - List open PRs
- `get_pull_request` - Get specific PR by number
- `list_pull_request_files` - List files in PR
- `create_pull_request` - Create new PR
- `read_file` - Read file from repository
- `create_file` - Create new file
- `update_file` - Update existing file
- `delete_file` - Delete file
- `overview_files_main_branch` - Get file overview
- `overview_files_current_branch` - Get current branch files

### Notion MCP Tools
- `list_notion_databases` - List databases with search
- `get_notion_database` - Get database details
- `query_notion_database` - Query database with filters
- `create_notion_page` - Create new page
- `get_notion_page` - Get page details
- `update_notion_page` - Update page properties
- `search_notion` - Search workspace

## Query Examples

### GitHub Queries
```python
# Get specific issue
result = workflow.run("Get issue #123 from microsoft/vscode")

# List open PRs
result = workflow.run("List all open pull requests in QwenLM/Qwen")

# Search for closed PRs
result = workflow.run("Find all closed pull requests in microsoft/vscode")
```

### Notion Queries
```python
# List databases
result = workflow.run("Show me all Notion databases")

# Query specific database
result = workflow.run("Show entries in the Tasks database")

# Get database details
result = workflow.run("Get details of the DP Interview Prep database")
```

### Mixed Queries
```python
# GitHub + Notion
result = workflow.run("Get issue #123 and create a Notion page with the details")
```

## Benefits of MCP Architecture

1. **Modularity**: Each service runs in its own process
2. **Scalability**: Servers can be scaled independently  
3. **Reliability**: Server failures don't crash the main application
4. **Security**: Better isolation of credentials and operations
5. **Extensibility**: Easy to add new services as MCP servers
6. **Debugging**: Each service can be monitored independently

## Troubleshooting

### MCP Servers Not Starting
1. Check environment variables are set correctly
2. Ensure ports 3001 and 3002 are available
3. Verify GitHub/Notion credentials are valid

### Connection Errors
1. Ensure MCP servers are running
2. Check firewall/network settings
3. Verify server ports in configuration

### Tool Execution Failures
1. Check server logs for detailed error messages
2. Verify API credentials have proper permissions
3. Ensure repository/database access rights

## Development Notes

### Adding New MCP Tools

To add a new GitHub tool:
1. Add tool definition to `github_mcp_server.py` in `list_tools()`
2. Implement tool logic in `_execute_tool()`
3. Update planner prompts to include new tool

To add a new Notion tool:
1. Add tool definition to `notion_mcp_server.py` in `list_tools()`
2. Implement tool logic in `_execute_tool()`
3. Update planner prompts to include new tool

### Backward Compatibility

The old tool implementations are still present but not used. The `parent_agent.py` has been updated to use MCP components while maintaining the same external interface.

## Migration Path

This represents a complete migration from direct tool calls to MCP architecture. The change is transparent to end users but provides significant architectural benefits for maintenance and scaling.
