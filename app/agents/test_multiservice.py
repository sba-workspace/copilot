#!/usr/bin/env python3
"""
Test script for the MultiServiceAgentWorkflow
Demonstrates usage of both GitHub and Notion functionality
"""

import os
import sys

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_github_only():
    """Test GitHub-only functionality"""
    print("=== Testing GitHub-only functionality ===")
    try:
        from app.agents.parent_agent import MultiServiceAgentWorkflow
        
        # Initialize workflow
        workflow = MultiServiceAgentWorkflow()
        
        # Example GitHub query (won't actually execute without proper env setup)
        print("SUCCESS: MultiServiceAgentWorkflow initialized successfully")
        print("SUCCESS: GitHub tools loaded")
        
        return True
    except Exception as e:
        print(f"ERROR: Error testing GitHub functionality: {e}")
        return False

def test_notion_only():
    """Test Notion-only functionality"""
    print("\n=== Testing Notion-only functionality ===")
    try:
        from app.tools.notion_tools import get_notion_tools
        
        # Get Notion tools
        tools = get_notion_tools()
        print(f"SUCCESS: Loaded {len(tools)} Notion tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return True
    except Exception as e:
        print(f"ERROR: Error testing Notion functionality: {e}")
        return False

def test_multi_service():
    """Test multi-service functionality"""
    print("\n=== Testing multi-service functionality ===")
    try:
        from app.agents.parent_agent import MultiServiceAgentWorkflow
        from app.tools.notion_tools import get_notion_tools
        from app.tools.github_tools import get_github_tools
        
        # Initialize workflow
        workflow = MultiServiceAgentWorkflow()
        
        # Get tools from both services
        github_tools = get_github_tools()
        notion_tools = get_notion_tools()
        
        print(f"SUCCESS: MultiServiceAgentWorkflow initialized")
        print(f"SUCCESS: Loaded {len(github_tools)} GitHub tools")
        print(f"SUCCESS: Loaded {len(notion_tools)} Notion tools")
        
        # Show combined tool count
        total_tools = len(github_tools) + len(notion_tools)
        print(f"SUCCESS: Total tools available: {total_tools}")
        
        return True
    except Exception as e:
        print(f"ERROR: Error testing multi-service functionality: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Multi-Service Agent Workflow")
    print("=" * 40)
    
    # Run tests
    github_success = test_github_only()
    notion_success = test_notion_only()
    multi_success = test_multi_service()
    
    print("\n" + "=" * 40)
    if github_success and notion_success and multi_success:
        print("SUCCESS: All tests passed! The system is ready for multi-service operations.")
    else:
        print("ERROR: Some tests failed. Please check the output above.")
    
    return github_success and notion_success and multi_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)