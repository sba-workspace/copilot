import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_notion_tools():
    """Test that Notion tools can be initialized"""
    try:
        # Try to get Notion tools
        from app.tools.notion_tools import get_notion_tools
        tools = get_notion_tools()
        print(f"Successfully loaded {len(tools)} Notion tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        return True
    except Exception as e:
        print(f"Error loading Notion tools: {e}")
        return False

if __name__ == "__main__":
    print("Testing Notion tools...")
    success = test_notion_tools()
    if success:
        print("SUCCESS: Notion tools test passed")
    else:
        print("FAILURE: Notion tools test failed")