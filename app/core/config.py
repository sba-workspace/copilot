# app/core/config.py
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from the .env file.
# This ensures that Pydantic will find the variables regardless of the
# current working directory. The path is relative to this config.py file.
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """
    Pydantic settings class that loads configuration from environment variables.
    The field names must match the names in the .env file exactly.
    """
    # LLM Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY= Optional[str] = None
    # Weaviate Configuration
    WEAVIATE_URL: Optional[str] = None
    WEAVIATE_API_KEY: Optional[str] = None

    # Supabase Configuration
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_JWT_SECRET: Optional[str] = None

    # Notion API
    NOTION_API_KEY: Optional[str] = None

    # GitHub API
    GITHUB_API_TOKEN: Optional[str] = None
    GITHUB_APP_ID: Optional[str] = None
    GITHUB_APP_PRIVATE_KEY: Optional[str] = None
    GITHUB_REPOSITORY: Optional[str] = None

    # Slack API
    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_APP_TOKEN: Optional[str] = None

    # Redis configuration
    REDIS_URL: Optional[str] = None

    # Cohere API
    COHERE_API_KEY: Optional[str] = None

    # LangSmith configuration
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_TRACING: Optional[bool] = None
    LANGSMITH_ENDPOINT: Optional[str] = None
    LANGSMITH_PROJECT_NAME: Optional[str] = None

    # GPU configuration
    USE_GPU: Optional[bool] = None

    # FastAPI settings
    DEBUG: Optional[bool] = None
    APP_ENV: Optional[str] = None
    API_PREFIX: Optional[str] = None
    
    # Pydantic model configuration
    # Note: env_file is now redundant since we're loading it manually,
    # but it can be kept for clarity or as a fallback.
    # extra="ignore" is good practice to prevent future errors if new env vars are added.
    class Config:
        env_file = '.env'
        extra = 'ignore'

# Instantiate the settings object
settings = Settings()