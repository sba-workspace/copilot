# Create a simple test script (test_env.py)
import os
from dotenv import load_dotenv

load_dotenv()

print("GITHUB_API_TOKEN:", "Set" if os.getenv("GITHUB_API_TOKEN") else "Not set")
print("GITHUB_REPO_URL:", os.getenv("GITHUB_REPO_URL", "Not set"))
print("GOOGLE_API_KEY:", "Set" if os.getenv("GOOGLE_API_KEY") else "Not set")