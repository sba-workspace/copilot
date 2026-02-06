# MCP Server Configuration
# This file contains configuration for both GitHub and Notion MCP servers

import os
import logging
from pathlib import Path

# Server Configuration
GITHUB_MCP_PORT = int(os.getenv("GITHUB_MCP_PORT", "3001"))
NOTION_MCP_PORT = int(os.getenv("NOTION_MCP_PORT", "3002"))

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

# Notion Configuration
NOTION_API_KEY = os.getenv("NOTION_API_KEY")

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def validate_environment():
    """Validate that required environment variables are set"""
    errors = []
    
    if not GITHUB_TOKEN:
        errors.append("GITHUB_TOKEN not set")
    
    if not NOTION_API_KEY:
        errors.append("NOTION_API_KEY not set")
    
    if not GITHUB_REPOSITORY and GITHUB_TOKEN:
        errors.append("GITHUB_REPOSITORY not set (required when using GitHub)")
    
    if errors:
        logging.error("Environment validation failed:")
        for error in errors:
            logging.error(f"  - {error}")
        return False
    
    logging.info("Environment validation passed")
    return True

def get_server_config():
    """Get server configuration"""
    return {
        "github": {
            "host": "localhost",
            "port": GITHUB_MCP_PORT,
            "enabled": bool(GITHUB_TOKEN)
        },
        "notion": {
            "host": "localhost", 
            "port": NOTION_MCP_PORT,
            "enabled": bool(NOTION_API_KEY)
        }
    }
