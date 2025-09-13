"""
GitHub MCP Server
Provides GitHub functionality through Model Context Protocol
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
load_dotenv()

# MCP imports will be added once the package is installed
# For now, we'll use placeholders
try:
    from mcp import Server
    from mcp import InitializationOptions
    from mcp import (
        CallToolRequest,
        CallToolResult,
        ListToolsRequest, 
        ListToolsResult,
        Tool,
        TextContent
    )
except ImportError:
    # Placeholder classes for development
    class Server:
        def __init__(self, name): 
            self.name = name
        def list_tools(self): 
            def decorator(func):
                return func
            return decorator
        def call_tool(self): 
            def decorator(func):
                return func
            return decorator
    
    class Tool:
        def __init__(self, **kwargs): 
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ListToolsResult:
        def __init__(self, tools): self.tools = tools
    
    class CallToolResult:
        def __init__(self, content): self.content = content
    
    class TextContent:
        def __init__(self, type, text): self.type = type; self.text = text
from pydantic import BaseModel
import httpx
from github import Github, GithubException
from github.Repository import Repository

logger = logging.getLogger(__name__)

class GitHubMCPServer:
    """GitHub MCP Server implementation"""
    
    def __init__(self):
        self.app = Server("github-mcp")
        self.github_client: Optional[Github] = None
        self.repository: Optional[Repository] = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP handlers"""
        
        @self.app.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available GitHub tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="get_issues",
                        description="Get open issues from the repository",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    ),
                    Tool(
                        name="get_issue", 
                        description="Get a specific issue by number",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "issue_number": {"type": "integer", "description": "Issue number"}
                            },
                            "required": ["issue_number"]
                        }
                    ),
                    Tool(
                        name="comment_on_issue",
                        description="Add a comment to an issue",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "issue_number": {"type": "integer", "description": "Issue number"},
                                "comment": {"type": "string", "description": "Comment text"}
                            },
                            "required": ["issue_number", "comment"]
                        }
                    ),
                    Tool(
                        name="search_issues_and_prs",
                        description="Search issues and pull requests",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="list_open_pull_requests",
                        description="List open pull requests",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    ),
                    Tool(
                        name="get_pull_request",
                        description="Get a specific pull request by number", 
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "pr_number": {"type": "integer", "description": "Pull request number"}
                            },
                            "required": ["pr_number"]
                        }
                    ),
                    Tool(
                        name="list_pull_request_files",
                        description="List files changed in a pull request",
                        inputSchema={
                            "type": "object", 
                            "properties": {
                                "pr_number": {"type": "integer", "description": "Pull request number"}
                            },
                            "required": ["pr_number"]
                        }
                    ),
                    Tool(
                        name="create_pull_request",
                        description="Create a new pull request",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "PR title"},
                                "body": {"type": "string", "description": "PR description"},
                                "head": {"type": "string", "description": "Source branch"},
                                "base": {"type": "string", "description": "Target branch"}
                            },
                            "required": ["title", "head", "base"]
                        }
                    ),
                    Tool(
                        name="read_file",
                        description="Read a file from the repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string", "description": "Path to the file"}
                            },
                            "required": ["file_path"]
                        }
                    ),
                    Tool(
                        name="create_file",
                        description="Create a new file in the repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string", "description": "Path for the new file"},
                                "content": {"type": "string", "description": "File content"},
                                "commit_message": {"type": "string", "description": "Commit message"}
                            },
                            "required": ["file_path", "content", "commit_message"]
                        }
                    ),
                    Tool(
                        name="update_file",
                        description="Update an existing file in the repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string", "description": "Path to the file"},
                                "content": {"type": "string", "description": "New file content"},
                                "commit_message": {"type": "string", "description": "Commit message"}
                            },
                            "required": ["file_path", "content", "commit_message"]
                        }
                    ),
                    Tool(
                        name="delete_file",
                        description="Delete a file from the repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string", "description": "Path to the file"},
                                "commit_message": {"type": "string", "description": "Commit message"}
                            },
                            "required": ["file_path", "commit_message"]
                        }
                    ),
                    Tool(
                        name="overview_files_main_branch",
                        description="Get overview of files in main branch",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    ),
                    Tool(
                        name="overview_files_current_branch",
                        description="Get overview of files in current branch",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    )
                ]
            )
        
        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Execute a GitHub tool"""
            try:
                if not self._ensure_github_client():
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="GitHub client not initialized. Please check your configuration."
                        )]
                    )
                
                result = await self._execute_tool(name, arguments)
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text=json.dumps(result, default=str, indent=2)
                    )]
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error executing tool {name}: {str(e)}"
                    )]
                )

    def _ensure_github_client(self) -> bool:
        """Ensure GitHub client is initialized"""
        if self.github_client is None:
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                logger.error("GITHUB_TOKEN not found in environment")
                return False
            
            self.github_client = Github(github_token)
            
            # Get repository from environment
            repo_name = os.getenv("GITHUB_REPOSITORY")
            if repo_name:
                try:
                    self.repository = self.github_client.get_repo(repo_name)
                except GithubException as e:
                    logger.error(f"Failed to get repository {repo_name}: {str(e)}")
                    return False
        
        return True
    
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific GitHub tools"""
        
        if name == "get_issues":
            return await self._get_issues()
        elif name == "get_issue":
            return await self._get_issue(arguments["issue_number"])
        elif name == "comment_on_issue":
            return await self._comment_on_issue(arguments["issue_number"], arguments["comment"])
        elif name == "search_issues_and_prs":
            return await self._search_issues_and_prs(arguments["query"])
        elif name == "list_open_pull_requests":
            return await self._list_open_pull_requests()
        elif name == "get_pull_request":
            return await self._get_pull_request(arguments["pr_number"])
        elif name == "list_pull_request_files":
            return await self._list_pull_request_files(arguments["pr_number"])
        elif name == "create_pull_request":
            return await self._create_pull_request(
                arguments["title"],
                arguments.get("body", ""),
                arguments["head"],
                arguments["base"]
            )
        elif name == "read_file":
            return await self._read_file(arguments["file_path"])
        elif name == "create_file":
            return await self._create_file(
                arguments["file_path"],
                arguments["content"], 
                arguments["commit_message"]
            )
        elif name == "update_file":
            return await self._update_file(
                arguments["file_path"],
                arguments["content"],
                arguments["commit_message"]
            )
        elif name == "delete_file":
            return await self._delete_file(arguments["file_path"], arguments["commit_message"])
        elif name == "overview_files_main_branch":
            return await self._overview_files_main_branch()
        elif name == "overview_files_current_branch":
            return await self._overview_files_current_branch()
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _get_issues(self) -> Dict[str, Any]:
        """Get open issues"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            issues = self.repository.get_issues(state="open")
            # Filter out pull requests
            issues_list = [issue for issue in issues if not issue.pull_request]
            
            result = []
            for issue in issues_list[:20]:  # Limit to 20 issues
                result.append({
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body[:500] if issue.body else "",
                    "state": issue.state,
                    "created_at": issue.created_at.isoformat(),
                    "updated_at": issue.updated_at.isoformat(),
                    "user": issue.user.login if issue.user else None,
                    "url": issue.html_url
                })
            
            return {"issues": result, "count": len(result)}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_issue(self, issue_number: int) -> Dict[str, Any]:
        """Get specific issue"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            issue = self.repository.get_issue(number=issue_number)
            
            # Get comments
            comments = []
            for comment in issue.get_comments()[:10]:  # Limit to 10 comments
                comments.append({
                    "body": comment.body,
                    "user": comment.user.login if comment.user else None,
                    "created_at": comment.created_at.isoformat()
                })
            
            return {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "state": issue.state,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "user": issue.user.login if issue.user else None,
                "comments": comments,
                "url": issue.html_url
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _comment_on_issue(self, issue_number: int, comment: str) -> Dict[str, Any]:
        """Add comment to issue"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            issue = self.repository.get_issue(number=issue_number)
            issue_comment = issue.create_comment(comment)
            
            return {
                "success": True,
                "comment_id": issue_comment.id,
                "comment_url": issue_comment.html_url,
                "message": f"Comment added to issue #{issue_number}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _search_issues_and_prs(self, query: str) -> Dict[str, Any]:
        """Search issues and pull requests"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            # Add repo constraint to query if not present
            if "repo:" not in query:
                query = f"repo:{self.repository.full_name} {query}"
            
            issues = self.github_client.search_issues(query)
            
            result = []
            for issue in issues[:20]:  # Limit results
                result.append({
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body[:500] if issue.body else "",
                    "state": issue.state,
                    "is_pull_request": bool(issue.pull_request),
                    "created_at": issue.created_at.isoformat(),
                    "user": issue.user.login if issue.user else None,
                    "url": issue.html_url
                })
            
            return {"results": result, "count": len(result)}
        except Exception as e:
            return {"error": str(e)}
    
    async def _list_open_pull_requests(self) -> Dict[str, Any]:
        """List open pull requests"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            prs = self.repository.get_pulls(state="open")
            
            result = []
            for pr in prs[:20]:  # Limit to 20 PRs
                result.append({
                    "number": pr.number,
                    "title": pr.title,
                    "body": pr.body[:500] if pr.body else "",
                    "state": pr.state,
                    "created_at": pr.created_at.isoformat(),
                    "updated_at": pr.updated_at.isoformat(),
                    "user": pr.user.login if pr.user else None,
                    "head": pr.head.ref,
                    "base": pr.base.ref,
                    "url": pr.html_url
                })
            
            return {"pull_requests": result, "count": len(result)}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_pull_request(self, pr_number: int) -> Dict[str, Any]:
        """Get specific pull request"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            pr = self.repository.get_pull(number=pr_number)
            
            # Get comments
            comments = []
            for comment in pr.get_issue_comments()[:10]:
                comments.append({
                    "body": comment.body,
                    "user": comment.user.login if comment.user else None,
                    "created_at": comment.created_at.isoformat()
                })
            
            # Get commits
            commits = []
            for commit in pr.get_commits()[:10]:
                commits.append({
                    "sha": commit.sha,
                    "message": commit.commit.message,
                    "author": commit.commit.author.name if commit.commit.author else None,
                    "date": commit.commit.author.date.isoformat() if commit.commit.author else None
                })
            
            return {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body,
                "state": pr.state,
                "created_at": pr.created_at.isoformat(),
                "updated_at": pr.updated_at.isoformat(),
                "user": pr.user.login if pr.user else None,
                "head": pr.head.ref,
                "base": pr.base.ref,
                "comments": comments,
                "commits": commits,
                "url": pr.html_url
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _list_pull_request_files(self, pr_number: int) -> Dict[str, Any]:
        """List files in pull request"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            pr = self.repository.get_pull(number=pr_number)
            files = pr.get_files()
            
            result = []
            for file in files:
                result.append({
                    "filename": file.filename,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch[:1000] if file.patch else None  # Limit patch size
                })
            
            return {"files": result, "count": len(result)}
        except Exception as e:
            return {"error": str(e)}
    
    async def _create_pull_request(self, title: str, body: str, head: str, base: str) -> Dict[str, Any]:
        """Create pull request"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            pr = self.repository.create_pull(
                title=title,
                body=body,
                head=head,
                base=base
            )
            
            return {
                "success": True,
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "message": f"Pull request #{pr.number} created successfully"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Read file content"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            file_content = self.repository.get_contents(file_path)
            
            if isinstance(file_content, list):
                return {"error": f"{file_path} is a directory, not a file"}
            
            content = file_content.decoded_content.decode('utf-8')
            
            return {
                "file_path": file_path,
                "content": content,
                "size": file_content.size,
                "sha": file_content.sha
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _create_file(self, file_path: str, content: str, commit_message: str) -> Dict[str, Any]:
        """Create new file"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            result = self.repository.create_file(
                path=file_path,
                message=commit_message,
                content=content
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "commit_sha": result["commit"].sha,
                "message": f"File {file_path} created successfully"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _update_file(self, file_path: str, content: str, commit_message: str) -> Dict[str, Any]:
        """Update existing file"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            # Get current file to get SHA
            file_content = self.repository.get_contents(file_path)
            
            if isinstance(file_content, list):
                return {"error": f"{file_path} is a directory, not a file"}
            
            result = self.repository.update_file(
                path=file_path,
                message=commit_message,
                content=content,
                sha=file_content.sha
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "commit_sha": result["commit"].sha,
                "message": f"File {file_path} updated successfully"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _delete_file(self, file_path: str, commit_message: str) -> Dict[str, Any]:
        """Delete file"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            # Get current file to get SHA
            file_content = self.repository.get_contents(file_path)
            
            if isinstance(file_content, list):
                return {"error": f"{file_path} is a directory, not a file"}
            
            result = self.repository.delete_file(
                path=file_path,
                message=commit_message,
                sha=file_content.sha
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "commit_sha": result["commit"].sha,
                "message": f"File {file_path} deleted successfully"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _overview_files_main_branch(self) -> Dict[str, Any]:
        """Get file overview from main branch"""
        if not self.repository:
            return {"error": "Repository not configured"}
        
        try:
            contents = self.repository.get_contents("", ref=self.repository.default_branch)
            
            files = []
            self._collect_files(contents, files, "")
            
            return {
                "branch": self.repository.default_branch,
                "files": files[:50],  # Limit to 50 files
                "count": len(files)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _overview_files_current_branch(self) -> Dict[str, Any]:
        """Get file overview from current branch"""
        # For now, return main branch - could be enhanced to support active branch
        return await self._overview_files_main_branch()
    
    def _collect_files(self, contents, files: List[Dict[str, Any]], prefix: str = ""):
        """Recursively collect files from repository"""
        for content in contents:
            if content.type == "file":
                files.append({
                    "path": content.path,
                    "name": content.name,
                    "size": content.size,
                    "type": "file"
                })
            elif content.type == "dir":
                files.append({
                    "path": content.path,
                    "name": content.name,
                    "type": "directory"
                })
                # Recursively get directory contents (limit depth to avoid too many API calls)
                if len(prefix.split("/")) < 3:
                    try:
                        dir_contents = self.repository.get_contents(content.path)
                        self._collect_files(dir_contents, files, content.path)
                    except:
                        pass  # Skip if we can't access directory

    def run(self, host: str = "localhost", port: int = 3001):
        """Run the MCP server"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

# Entry point for running as standalone server
if __name__ == "__main__":
    server = GitHubMCPServer()
    server.run()
