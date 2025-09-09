from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from app.agents.parent_agent import MultiServiceAgentWorkflow
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


app = FastAPI(title="GitHub Agent Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check required environment variables for MCP
required_env_vars = ["GOOGLE_API_KEY"]  # Core requirement
github_vars = ["GITHUB_TOKEN", "GITHUB_REPOSITORY"] 
notion_vars = ["NOTION_API_KEY"]

missing_core = [var for var in required_env_vars if not os.getenv(var)]
if missing_core:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_core)}")

# Check service-specific variables
has_github = all(os.getenv(var) for var in github_vars)
has_notion = all(os.getenv(var) for var in notion_vars)

if not has_github and not has_notion:
    raise RuntimeError("At least one service (GitHub or Notion) must be configured")

print(f"Services enabled - GitHub: {has_github}, Notion: {has_notion}")

# Initialize the MCP-enabled multi-service agent workflow
workflow = MultiServiceAgentWorkflow()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    MCP-enabled chat endpoint for GitHub and Notion operations.
    The user provides a natural language query, and the agent routes it to appropriate MCP servers.
    Services are configured via environment variables.
    """
    try:
        result = workflow.run(request.query)
        if result["success"]:
            return {
                "response": result["response"],
                "success": True,
                "repo_info": result["repo_info"]
            }
        else:
            return {
                "response": result["response"],
                "success": False,
                "errors": result["errors"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)