#!/usr/bin/env python3
"""
Test script for MCP integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.parent_agent import MultiServiceAgentWorkflow
from app.mcp.mcp_client import initialize_mcp_clients

async def test_mcp_integration():
    """Test MCP integration"""
    print("Testing MCP Integration...")
    
    # Set test environment variables if not already set
    if not os.getenv("GITHUB_REPOSITORY"):
        os.environ["GITHUB_REPOSITORY"] = "microsoft/vscode"  # Default test repo
    
    print(f"Using GitHub repository: {os.getenv('GITHUB_REPOSITORY')}")
    
    # Initialize MCP clients
    print("\n1. Initializing MCP clients...")
    try:
        init_results = await initialize_mcp_clients()
        print(f"   GitHub MCP: {'✓' if init_results.get('github') else '✗'}")
        print(f"   Notion MCP: {'✓' if init_results.get('notion') else '✗'}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test workflow creation
    print("\n2. Creating MCP workflow...")
    try:
        workflow = MultiServiceAgentWorkflow()
        print("   ✓ Workflow created successfully")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    
    # Test GitHub query
    print("\n3. Testing GitHub query...")
    try:
        repo = os.getenv('GITHUB_REPOSITORY', 'microsoft/vscode')
        github_query = f"Get issue #1 from {repo}"
        print(f"   Query: {github_query}")
        
        result = workflow.run(github_query)
        
        if result.get("success"):
            print("   ✓ GitHub query completed successfully")
            if result.get("task_results"):
                print(f"   Found {len(result['task_results'])} task results")
                print(result['task_results'])
        else:
            print(f"   ✗ GitHub query failed: {result.get('response', 'Unknown error')}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test Notion query (if API key is available)
    print("\n4. Testing Notion query...")
    if os.getenv("NOTION_API_KEY"):
        try:
            notion_query = "List my items in product database"
            print(f"   Query: {notion_query}")
            
            result = workflow.run(notion_query)
            
            if result.get("success"):
                print("   ✓ Notion query completed successfully")
                if result.get("task_results"):
                    print(f"   Found {len(result['task_results'])} task results")
                    print(result["task_results"])
            else:
                print(f"   ✗ Notion query failed: {result.get('response', 'Unknown error')}")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
    else:
        print("   ⚠ Skipped (NOTION_API_KEY not set)")
    
    # Test mixed query
    print("\n5. Testing mixed service query...")
    try:
        repo = os.getenv('GITHUB_REPOSITORY', 'microsoft/vscode')
        mixed_query = f"Get issue #1 from {repo} and list my Notion databases"
        print(f"   Query: {mixed_query}")
        
        result = workflow.run(mixed_query)
        
        if result.get("success"):
            print("   ✓ Mixed query completed successfully")
            execution_plan = result.get("execution_plan", {})
            github_tasks = len(execution_plan.get("github", {}))
            notion_tasks = len(execution_plan.get("notion", {}))
            print(f"   GitHub tasks: {github_tasks}, Notion tasks: {notion_tasks}")
            print(result['task_results'])
        else:
            print(f"   ✗ Mixed query failed: {result.get('response', 'Unknown error')}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n6. Testing environment configuration...")
    required_vars = ["GITHUB_TOKEN", "GITHUB_REPOSITORY"]
    optional_vars = ["NOTION_API_KEY"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✓ {var}: {'*' * min(len(value), 10)}")
        else:
            print(f"   ✗ {var}: Not set")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✓ {var}: {'*' * min(len(value), 10)}")
        else:
            print(f"   ⚠ {var}: Not set (optional)")
    
    print("\n=== MCP Integration Test Complete ===")

def sync_test():
    """Synchronous wrapper for async test"""
    asyncio.run(test_mcp_integration())

if __name__ == "__main__":
    sync_test()


