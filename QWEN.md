# Project Context for Qwen Code

## Project Overview
This is an AI Task Copilot - an intelligent task management system that integrates with GitHub to streamline development workflows. The system uses LLMs and agentic AI patterns to plan, execute, and validate tasks based on natural language user queries.

## Core Architecture
The application follows a LangGraph-based agentic workflow with these components:

### 1. Main Application (FastAPI)
- Entry point: `app/main.py`
- Exposes a `/chat` endpoint for natural language queries
- Uses FastAPI with CORS middleware
- Requires environment variables: `GITHUB_API_TOKEN`, `GITHUB_REPOSITORY`, `GOOGLE_API_KEY`

### 2. Agent Workflow (`app/agents/`)
The system implements a multi-agent workflow using LangGraph:

#### Parent Agent (`parent_agent.py`)
- Orchestrates the entire workflow
- Creates a state graph with nodes for each specialized agent
- Manages repository context setup

#### Planner Agent (`planner.py`)
- Analyzes user queries to create execution plans
- Generates structured JSON plans with GitHub tool specifications
- Uses strict prompt engineering to ensure valid JSON output

#### Executor Agent (`executor.py`)
- Executes individual tasks from the plan
- Handles parameter extraction and mapping
- Manages GitHub API tool execution
- Maintains execution status tracking

#### Validator Agent (`validator.py`)
- Checks execution completion status
- Validates results and determines retry needs
- Implements retry logic with maximum attempts

#### Formatter Agent (`formatter.py`)
- Formats final responses for user consumption
- Generates natural language summaries of execution results
- Handles both informational and action-based queries

### 3. State Management (`utils.py`)
- Defines the `AgentState` TypedDict for shared state
- Contains task status enums and data models
- Maintains execution context across agents

## GitHub Integration
The system integrates with GitHub through specialized tools:
- Issue management (get, list, comment)
- Pull request operations (create, list, get files)
- File operations (read, create, update, delete)
- Repository search capabilities
- Branch operations

## Key Dependencies
- FastAPI for web framework
- LangChain/LangGraph for agentic workflows
- Google Generative AI (Gemini) for LLM capabilities
- PyGithub for GitHub API integration
- Pydantic for data validation

## Environment Variables Required
- `GITHUB_API_TOKEN`: GitHub personal access token
- `GITHUB_REPOSITORY`: Default repository in owner/repo format
- `GOOGLE_API_KEY`: Google API key for Gemini models

## Workflow Process
1. User sends natural language query to `/chat` endpoint
2. Planner agent creates structured execution plan
3. Executor agent processes tasks sequentially
4. Validator agent checks completion and validates results
5. Formatter agent generates natural language response
6. Response returned to user

## Error Handling
- Rate limit detection and handling
- Repository access validation
- Parameter validation and mapping
- Retry mechanisms with maximum attempts
- Graceful error reporting