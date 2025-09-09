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

# Check required environment variables
required_env_vars = ["GITHUB_API_TOKEN", "GITHUB_REPOSITORY", "GOOGLE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize the multi-service agent workflow
workflow = MultiServiceAgentWorkflow()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for GitHub operations.
    The user provides a natural language query, and the agent handles it.
    The repository is specified in the environment, so no repo URL is needed in the request.
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