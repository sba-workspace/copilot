# app/tools/github_tools.py
# Fixed GitHub tools for dynamic repository configuration

from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field, model_validator
from langchain.tools import BaseTool
from langchain_community.utilities.github import GitHubAPIWrapper
import logging
import os

logger = logging.getLogger(__name__)

# Global wrapper instance for reuse
_global_wrapper = None

def get_wrapper(github_api_wrapper: Optional[GitHubAPIWrapper] = None, repo: Optional[str] = None) -> GitHubAPIWrapper:
    """
    Return a GitHubAPIWrapper. If none provided, create a new one.
    Now supports dynamic repository configuration.
    """
    global _global_wrapper
    
    if github_api_wrapper is not None and repo is not None:
        current_repo = getattr(github_api_wrapper, "github_repository", None)
        if current_repo != repo:
            logger.info(f"Recreating wrapper for repository: {repo}")
            github_token = os.getenv("GITHUB_API_TOKEN")
            if not github_token:
                raise RuntimeError("GITHUB_API_TOKEN environment variable is not set")
            return GitHubAPIWrapper( github_repository=repo)
        return github_api_wrapper
    
    # If we have a repo parameter, create a wrapper with it
    if repo:
        try:
            github_token = os.getenv("GITHUB_API_TOKEN")
            if not github_token:
                raise RuntimeError("GITHUB_API_TOKEN environment variable is not set")
            wrapper = GitHubAPIWrapper( github_repository=repo)
            _global_wrapper = wrapper
            logger.info(f"Created GitHubAPIWrapper for repository: {repo}")
            return wrapper
        except Exception as e:
            logger.error(f"Failed to create wrapper for repo {repo}: {e}")
            raise RuntimeError(f"Could not create GitHubAPIWrapper instance for repo {repo}: {e}")
    
    # Try to reuse global wrapper if it exists
    if _global_wrapper is not None:
        logger.info(f"Reusing existing wrapper for: {getattr(_global_wrapper, 'github_repository', 'unknown')}")
        return _global_wrapper
    
    # Try environment variable for repository
    repo_env = os.getenv("GITHUB_REPOSITORY")
    if repo_env:
        try:
            github_token = os.getenv("GITHUB_API_TOKEN")
            if not github_token:
                raise RuntimeError("GITHUB_API_TOKEN environment variable is not set")
            wrapper = GitHubAPIWrapper( github_repository=repo_env)
            _global_wrapper = wrapper
            logger.info(f"Created GitHubAPIWrapper from env for repository: {repo_env}")
            return wrapper
        except Exception as e:
            logger.error(f"Failed to create wrapper from env: {e}")
            raise RuntimeError(f"Could not create GitHubAPIWrapper instance from env: {e}")
    
    # Check if we have a token but no repo specified
    github_token = os.getenv("GITHUB_API_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_API_TOKEN environment variable is not set")
    
    # Instead of creating a dummy wrapper, raise an error asking for repository configuration
    raise RuntimeError(
        "No GitHub repository specified. Please provide a repository either:\n"
        "1. As a parameter to get_github_tools(repo='owner/repo')\n"
        "2. Set the GITHUB_REPOSITORY environment variable\n"
        "3. Call configure_wrapper_repository('owner/repo') before using tools"
    )

def configure_wrapper_repository(repo: str) -> GitHubAPIWrapper:
    """
    Configure or create a wrapper with a specific repository.
    Returns the configured wrapper.
    """
    global _global_wrapper
    
    try:
        github_token = os.getenv("GITHUB_API_TOKEN")
        if not github_token:
            raise RuntimeError("GITHUB_API_TOKEN environment variable is not set")
        
        # Create new wrapper with the specified repository
        wrapper = GitHubAPIWrapper(github_repository=repo)
        _global_wrapper = wrapper
        logger.info(f"Configured wrapper repository to: {repo}")
        return wrapper
    except Exception as e:
        logger.error(f"Failed to configure wrapper with repo {repo}: {e}")
        raise RuntimeError(f"Could not configure GitHubAPIWrapper for repository {repo}: {e}")

def _get_repo(wrapper: GitHubAPIWrapper):
    """Return a PyGithub Repository instance, trying wrapper.github_repo_instance first."""
    if hasattr(wrapper, "github_repo_instance") and wrapper.github_repo_instance:
        return wrapper.github_repo_instance
    
    # Get repository name and validate it's properly configured
    repo_name = getattr(wrapper, "github_repository", None)
    if repo_name is None:
        raise RuntimeError("GitHub repository not configured on the wrapper.")
    
    try:
        return wrapper.github.get_repo(repo_name)
    except Exception as e:
        raise RuntimeError(f"Failed to access repository '{repo_name}': {e}")

def _serialize_issue(issue) -> Dict[str, Any]:
    """Safely serialize an issue object to a dictionary, handling all possible response types."""
    try:
        # Case 1: It's a string (likely an error message)
        if isinstance(issue, str):
            logger.error(f"Received string instead of issue object: {issue}")
            return {
                "error": issue,
                "type": "string_response",
                "is_valid": False
            }
        
        # Case 2: It's a dictionary with error information
        if isinstance(issue, dict) and ("error" in issue or "message" in issue):
            error_msg = issue.get("error", issue.get("message", "Unknown API error"))
            logger.error(f"API error response: {error_msg}")
            return {
                "error": error_msg,
                "status": issue.get("status"),
                "documentation_url": issue.get("documentation_url"),
                "type": "api_error",
                "is_valid": False
            }
        
        # Case 3: It's a dictionary with issue data
        if isinstance(issue, dict):
            return {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "body": issue.get("body", "")[:1000] if issue.get("body") else None,  # Limit body size
                "state": issue.get("state"),
                "url": issue.get("html_url") or issue.get("url"),
                "user": issue.get("user", {}).get("login") if issue.get("user") else None,
                "type": "dict",
                "is_valid": True
            }
        
        # Case 4: It's an object with attributes
        if hasattr(issue, "number"):
            # Handle GitHub issue objects
            user_login = None
            if hasattr(issue, "user") and issue.user:
                user_login = getattr(issue.user, "login", None)
            
            return {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body[:1000] if issue.body else None,  # Limit body size
                "state": issue.state,
                "url": getattr(issue, "html_url", None) or getattr(issue, "url", None),
                "user": user_login,
                "type": "github_object",
                "is_valid": True
            }
        
        # Unknown type
        logger.warning(f"Unexpected issue type: {type(issue)} - {str(issue)[:100]}")
        return {
            "error": f"Unsupported issue format: {type(issue)}",
            "raw": str(issue)[:500],
            "is_valid": False
        }

    except Exception as e:
        logger.exception(f"Error serializing issue: {e}")
        return {
            "error": f"Failed to serialize issue: {str(e)}",
            "is_valid": False
        }

def _serialize_pr(pr) -> Dict[str, Any]:
    """Safely serialize a pull request object to a dictionary."""
    try:
        # Handle string responses (error messages)
        if isinstance(pr, str):
            return {
                "error": pr,
                "type": "string_response",
                "is_valid": False
            }
        
        # Handle error dictionaries
        if isinstance(pr, dict) and ("error" in pr or "message" in pr):
            error_msg = pr.get("error", pr.get("message", "Unknown API error"))
            return {
                "error": error_msg,
                "status": pr.get("status"),
                "is_valid": False
            }
        
        # Handle dictionary responses
        if isinstance(pr, dict):
            return {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "body": pr.get("body", "")[:1000] if pr.get("body") else None,
                "state": pr.get("state"),
                "url": pr.get("html_url") or pr.get("url"),
                "head": pr.get("head", {}).get("ref") if pr.get("head") else None,
                "base": pr.get("base", {}).get("ref") if pr.get("base") else None,
                "user": pr.get("user", {}).get("login") if pr.get("user") else None,
                "status": pr.get("state"),
                "is_valid": True
            }
        
        # Handle GitHub PR objects
        if hasattr(pr, "number"):
            head_ref = None
            base_ref = None
            user_login = None
            
            if hasattr(pr, "head") and pr.head:
                head_ref = getattr(pr.head, "ref", None)
            if hasattr(pr, "base") and pr.base:
                base_ref = getattr(pr.base, "ref", None)
            if hasattr(pr, "user") and pr.user:
                user_login = getattr(pr.user, "login", None)
            
            return {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body[:1000] if pr.body else None,
                "state": pr.state,
                "url": getattr(pr, "html_url", None) or getattr(pr, "url", None),
                "head": head_ref,
                "base": base_ref,
                "user": user_login,
                "status": pr.state,
                "is_valid": True
            }
        
        # Unknown type
        logger.warning(f"Unexpected PR type: {type(pr)}")
        return {
            "error": "Unsupported PR format",
            "raw_type": str(type(pr)),
            "is_valid": False
        }
    except Exception as e:
        logger.exception(f"Error serializing PR: {e}")
        return {
            "error": f"Failed to serialize PR: {str(e)}",
            "is_valid": False
        }
        
def sanitize_diff_content(diff_text: str) -> str:
    """Sanitize diff content to prevent tokenizer special character issues"""
    if not diff_text or not isinstance(diff_text, str):
        return diff_text
    
    # Replace problematic characters with HTML entities
    sanitized = diff_text.replace(">", ">").replace("<", "<")
    
    # Handle common diff patterns that might cause issues
    sanitized = sanitized.replace("@@ ", "DIFF_HEADER ")
    sanitized = sanitized.replace("+++ ", "ADDED_FILE ")
    sanitized = sanitized.replace("--- ", "REMOVED_FILE ")
    
    return sanitized

    
def _list_files_in_repo_recursive(repo, path: str = "", ref: Optional[str] = None) -> List[str]:
    """
    Recursively list file paths in repo at given ref. Uses PyGithub repo.get_contents.
    """
    files: List[str] = []
    try:
        contents = repo.get_contents(path, ref=ref)
    except Exception as e:
        logger.error(f"Error accessing path '{path}': {e}")
        return files
    
    if not isinstance(contents, list):
        # single file
        return [contents.path]
    
    for entry in contents:
        try:
            if entry.type == "file":
                files.append(entry.path)
            elif entry.type == "dir":
                files.extend(_list_files_in_repo_recursive(repo, entry.path, ref=ref))
        except Exception as e:
            logger.error(f"Error processing entry '{entry.path}': {e}")
            continue
    
    return files

# Input schemas (keeping the same as before)
class NoInput(BaseModel):
    pass

class SearchIssuesInput(BaseModel):
    query: str = Field(..., description="Search query (GitHub search syntax).")
    per_page: Optional[int] = Field(30, description="Max results to return.")
    @model_validator(mode='after')
    def ensure_valid_query(self) -> 'SearchIssuesInput':
        # If the query looks like natural language, try to convert it
        query = self.query.lower()
        
        if "list" in query or "show me" in query:
            # This is likely a natural language request, not a GitHub query
            if "closed" in query and "pull request" in query:
                self.query = "is:pr is:closed"
            elif "open" in query and "pull request" in query:
                self.query = "is:pr is:open"
            elif "issue" in query and "closed" in query:
                self.query = "is:issue is:closed"
            elif "issue" in query and "open" in query:
                self.query = "is:issue is:open"
        
        # Ensure repository context is included
        if "repo:" not in self.query:
            repo_owner = os.getenv("GITHUB_REPO_URL", "").split("/")[0]
            repo_name = os.getenv("GITHUB_REPO_URL", "").split("/")[1] if len(os.getenv("GITHUB_REPO_URL", "").split("/")) > 1 else ""
            if repo_owner and repo_name:
                self.query = f"repo:{repo_owner}/{repo_name} {self.query}"
                
        return self
class GetIssueInput(BaseModel):
    issue_number: int = Field(..., description="Issue number (integer).")

class CommentIssueInput(BaseModel):
    issue_number: int = Field(..., description="Issue number (integer).")
    comment: str = Field(..., description="Comment body to post.")
    
    @model_validator(mode='before')
    @classmethod
    def ensure_comment_exists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "comment" not in data and "text" in data:
                data["comment"] = data["text"]
            elif "comment" not in data and "description" in data:
                data["comment"] = data["description"]
        return data

class ListPRsInput(BaseModel):
    state: Optional[str] = Field("open", description="open/closed/all")
    per_page: Optional[int] = Field(30)

class GetPRInput(BaseModel):
    pr_number: int = Field(..., description="Pull Request number (integer).")

class CreatePRInput(BaseModel):
    title: str
    body: Optional[str] = None
    head: str = Field(..., description="Head branch (source).")
    base: str = Field(..., description="Base branch (target).")

class PRFilesInput(BaseModel):
    pr_number: int = Field(..., description="PR number.")

class CreateFileInput(BaseModel):
    path: str = Field(..., description="Repo path for file (e.g. src/app.py)")
    message: str = Field(..., description="Commit message")
    content: str = Field(..., description="File content (string)")
    branch: Optional[str] = Field(None, description="Branch to create file in. If None uses default branch.")

class ReadFileInput(BaseModel):
    path: str = Field(..., description="File path to read")
    ref: Optional[str] = Field(None, description="Branch, tag, or commit SHA to read from (optional)")

class UpdateFileInput(BaseModel):
    path: str = Field(..., description="File path to update")
    message: str = Field(..., description="Commit message")
    content: str = Field(..., description="New file content")
    branch: Optional[str] = Field(None, description="Branch")
    sha: Optional[str] = Field(None, description="Existing file SHA (optional; auto-fetched if omitted).")

class DeleteFileInput(BaseModel):
    path: str = Field(..., description="File path to delete")
    message: str = Field(..., description="Commit message")
    branch: Optional[str] = Field(None, description="Branch")
    sha: Optional[str] = Field(None, description="Existing file SHA (optional; auto-fetched if omitted).")

class OverviewFilesInput(BaseModel):
    branch: Optional[str] = Field(None, description="Branch name to list files from (defaults to default branch).")

# Tool implementations (keeping the same structure)
class GetIssuesTool(BaseTool):
    name: str = "get_issues"
    description: str = "List repository issues (default: open). No inputs required."
    args_schema: type[BaseModel] = NoInput

    def _run(self, max_issues: int = 10) -> List[Dict[str, Any]]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            
            try:
                # Get issues with limited pagination
                issues = repo.get_issues(state="open")
                
                # Check if we got string responses instead of issues
                if issues and isinstance(issues[0], str):
                    logger.error(f"GitHub API returned string response: {issues[0]}")
                    return [{
                        "error": f"GitHub API error: {issues[0]}",
                        "is_valid": False
                    }]
                
                # Process issues with validation
                valid_issues = []
                for i, issue in enumerate(issues):
                    if i >= max_issues:
                        break
                    
                    serialized = _serialize_issue(issue)
                    if serialized.get("is_valid", False):
                        valid_issues.append(serialized)
                    else:
                        logger.warning(f"Issue serialization failed: {serialized.get('error', 'Unknown error')}")
                
                if not valid_issues:
                    return [{
                        "error": "No valid issues found in the repository",
                        "is_valid": False
                    }]
                
                return valid_issues
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"GitHub API error: {error_msg}")
                
                # Check for common GitHub API errors
                if "403" in error_msg and "rate limit" in error_msg.lower():
                    return [{
                        "error": "GitHub API rate limit exceeded. Please try again later.",
                        "is_valid": False,
                        "rate_limit_exceeded": True
                    }]
                elif "404" in error_msg:
                    return [{
                        "error": "Repository not found. Please check the repository name and your permissions.",
                        "is_valid": False
                    }]
                
                return [{
                    "error": f"Failed to retrieve issues: {error_msg}",
                    "is_valid": False
                }]
                
        except Exception as e:
            logger.exception(f"Critical error in get_issues: {str(e)}")
            return [{
                "error": f"Critical system error: {str(e)}",
                "is_valid": False
            }]

    async def _arun(self):
        raise NotImplementedError("Async not implemented.")
    
    
class GetIssueTool(BaseTool):
    name: str = "get_issue"
    description: str = "Get details and comments of a specific issue. Input: issue_number (int)."
    args_schema: type[BaseModel] = GetIssueInput

    def _run(self, issue_number: int) -> Dict[str, Any]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            
            try:
                issue = repo.get_issue(number=issue_number)
            except Exception as e:
                # Handle 404 and other GitHub API errors
                if "404" in str(e):
                    return {
                        "error": f"Issue #{issue_number} not found in the repository",
                        "number": issue_number,
                        "exists": False
                    }
                raise
            
            # Get comments safely
            try:
                comments = [c.body for c in issue.get_comments()]
            except Exception as e:
                logger.warning(f"Could not fetch comments for issue #{issue_number}: {e}")
                comments = []
            
            result = _serialize_issue(issue)
            result["comments"] = comments
            result["exists"] = True
            return result
            
        except Exception as e:
            logger.error(f"Error getting issue #{issue_number}: {e}")
            return {
                "error": f"Failed to get issue #{issue_number}: {str(e)}",
                "number": issue_number,
                "exists": False
            }
            
            
class CommentOnIssueTool(BaseTool):
    name: str = "comment_on_issue"
    description: str = "Post a comment to an issue. Input: issue_number (int) and comment (str)."
    args_schema: type[BaseModel] = CommentIssueInput

    github_api_wrapper: Optional[GitHubAPIWrapper] = None

    def __init__(self, github_api_wrapper: Optional[GitHubAPIWrapper] = None, **kwargs):
        super().__init__(**kwargs)
        self.github_api_wrapper = github_api_wrapper

    def _run(self, issue_number: int, comment: str) -> Dict[str, Any]:
        """Always use PyGithub API directly to avoid wrapper inconsistencies"""
        try:
            wrapper = self.github_api_wrapper or get_wrapper()
            repo = _get_repo(wrapper)
            
            issue = repo.get_issue(number=issue_number)
            posted = issue.create_comment(comment)
            return {
                "result": "ok", 
                "comment_url": getattr(posted, "html_url", None),
                "issue_number": issue_number
            }
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg and "Resource not accessible by integration" in error_msg:
                raise RuntimeError(
                    f"Permission denied: The GitHub App does not have permission to comment on issues in this repository. "
                    f"Please ensure the app is installed on the repository and has 'issues: write' permissions."
                )
            else:
                raise RuntimeError(f"Failed to comment on issue #{issue_number}: {error_msg}")

    async def _arun(self, issue_number: int, comment: str):
        raise NotImplementedError("Async not implemented.")

class SearchIssuesAndPRsTool(BaseTool):
    name: str = "search_issues_and_prs"
    description: str = "Search issues and pull requests using GitHub search syntax. Input: query."
    args_schema: type[BaseModel] = SearchIssuesInput

    def _run(self, query: str, per_page: int = 30) -> List[Dict[str, Any]]:
        try:
            wrapper = get_wrapper()
            if hasattr(wrapper, "search_issues"):
                results = wrapper.search_issues(query=query)
                try:
                    return [_serialize_issue(r) for r in results][:per_page]
                except Exception:
                    return list(results)[:per_page]
            
            gh = wrapper.github
            items = gh.search_issues(query)
            out: List[Dict[str, Any]] = []
            for i, item in enumerate(items):
                if i >= per_page:
                    break
                out.append(_serialize_issue(item))
            return out
        except Exception as e:
            logger.error(f"Error searching issues with query '{query}': {e}")
            raise RuntimeError(f"Failed to search issues: {e}")

    async def _arun(self, query: str, per_page: int = 30):
        raise NotImplementedError("Async not implemented.")

class ListOpenPRsTool(BaseTool):
    name: str = "list_open_pull_requests"
    description: str = "List open pull requests (default open)."
    args_schema: type[BaseModel] = ListPRsInput

    def _run(self, state: str = "open", per_page: int = 10) -> List[Dict[str, Any]]:
        """List open pull requests with proper pagination and limited results"""
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            
            # Use a reasonable default for per_page (5 as requested)
            # Cap at 30 to avoid GitHub API limits
            per_page = max(1, min(per_page, 10))
            
            # Get only the first page with the requested number of PRs
            pulls = repo.get_pulls(state=state)
            
            # Convert to our format without triggering additional API calls
            pr_list = []
            # Iterate through the paginated list and limit results
            for i, pr in enumerate(pulls):
                if i >= per_page:
                    break
                pr_list.append(_serialize_pr(pr))
                
                try:
                    serialized = _serialize_pr(pr)
                    if serialized.get("is_valid", False):
                        pr_list.append(serialized)
                        
                    else:
                        logger.debug(f"Skipping invalid PR: {serialized.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.warning(f"Error processing PR: {str(e)}")
                    continue
            
            if not pr_list:
                return [{
                    "error": "No pull requests found matching your criteria",
                    "is_valid": False
                }]
            
            return pr_list
        except Exception as e:
            logger.error(f"Error fetching pull requests: {str(e)}")
            raise RuntimeError(f"Failed to fetch pull requests: {str(e)}")

    async def _arun(self, state: str = "open", per_page: int = 5):
        raise NotImplementedError("Async not implemented.")

class GetPullRequestTool(BaseTool):
    name: str = "get_pull_request"
    description: str = "Get details of a pull request (number)."
    args_schema: type[BaseModel] = GetPRInput

    def _run(self, pr_number: int) -> Dict[str, Any]:
        try:
            wrapper = get_wrapper()
            if hasattr(wrapper, "get_pull_request"):
                pr = wrapper.get_pull_request(pr_number)
                try:
                    return _serialize_pr(pr)
                except Exception:
                    return pr
            
            repo = _get_repo(wrapper)
            pr = repo.get_pull(pr_number)
            result = _serialize_pr(pr)
            result["commits"] = [c.sha for c in pr.get_commits()]
            result["files"] = [{"filename": f.filename, "changes": f.changes, "status": f.status} for f in pr.get_files()]
            return result
        except Exception as e:
            logger.error(f"Error getting PR #{pr_number}: {e}")
            raise RuntimeError(f"Failed to get pull request #{pr_number}: {e}")

    async def _arun(self, pr_number: int):
        raise NotImplementedError("Async not implemented.")

class CreatePullRequestTool(BaseTool):
    name: str = "create_pull_request"
    description: str = "Create a pull request. Input: title, body, head, base."
    args_schema: type[BaseModel] = CreatePRInput

    def _run(self, title: str, head: str, base: str, body: Optional[str] = None) -> Dict[str, Any]:
        try:
            wrapper = get_wrapper()
            if hasattr(wrapper, "create_pull_request"):
                pr = wrapper.create_pull_request(title=title, body=body, head=head, base=base)
                try:
                    return _serialize_pr(pr)
                except Exception:
                    return {"result": pr}
            
            repo = _get_repo(wrapper)
            pr = repo.create_pull(title=title, body=body, head=head, base=base)
            return _serialize_pr(pr)
        except Exception as e:
            logger.error(f"Error creating PR: {e}")
            raise RuntimeError(f"Failed to create pull request: {e}")

    async def _arun(self, title: str, head: str, base: str, body: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class ListPullRequestFilesTool(BaseTool):
    name: str = "list_pull_request_files"
    description: str = "List files changed in a pull request. Input: pr_number (int)."
    args_schema: type[BaseModel] = PRFilesInput

    def _run(self, pr_number: int) -> List[Dict[str, Any]]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            
            # Get the pull request
            try:
                pr = repo.get_pull(pr_number)
            except Exception as e:
                if "404" in str(e):
                    return [{
                        "error": f"Pull Request #{pr_number} not found in the repository",
                        "is_valid": False
                    }]
                raise
            
            # Get files with proper error handling
            try:
                files = pr.get_files()
            except Exception as e:
                logger.error(f"Error fetching files for PR #{pr_number}: {str(e)}")
                return [{
                    "error": f"Could not retrieve files for PR #{pr_number}: {str(e)}",
                    "is_valid": False
                }]
            
            # Process files with sanitization
            result = []
            for f in files:
                # Create a sanitized version of the patch
                patch = getattr(f, "patch", "")
                if patch:
                    sanitized_patch = sanitize_diff_content(patch)
                else:
                    sanitized_patch = ""
                
                result.append({
                    "filename": sanitize_diff_content(f.filename),
                    "status": sanitize_diff_content(f.status),
                    "changes": f.changes,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "raw_url": getattr(f, "raw_url", ""),
                    "blob_url": getattr(f, "blob_url", ""),
                    "patch": sanitized_patch
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing PR files for #{pr_number}: {e}")
            return [{
                "error": f"Failed to list files for pull request #{pr_number}: {str(e)}",
                "is_valid": False
            }]
            
            
class CreateFileTool(BaseTool):
    name: str = "create_file"
    description: str = "Create a file in the repository. Input: path, message, content, branch (optional)."
    args_schema: type[BaseModel] = CreateFileInput

    def _run(self, path: str, message: str, content: str, branch: Optional[str] = None) -> Dict[str, Any]:
        try:
            wrapper = get_wrapper()
            if hasattr(wrapper, "create_file"):
                return wrapper.create_file(path=path, message=message, content=content, branch=branch)
            
            repo = _get_repo(wrapper)
            created = repo.create_file(path, message, content, branch=branch)
            return {"commit_sha": getattr(created[0], "sha", None), "file": getattr(created[1], "path", None)}
        except Exception as e:
            logger.error(f"Error creating file '{path}': {e}")
            raise RuntimeError(f"Failed to create file '{path}': {e}")

    async def _arun(self, path: str, message: str, content: str, branch: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read a file's contents. Input: path (and optionally ref for branch/commit)."
    args_schema: type[BaseModel] = ReadFileInput

    def _run(self, path: str, ref: Optional[str] = None) -> str:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            
            contents = repo.get_contents(path, ref=ref)
            
            if hasattr(contents, "content"):
                import base64
                return base64.b64decode(contents.content).decode('utf-8')
            return str(contents)
        except Exception as e:
            logger.error(f"Error reading file '{path}': {e}")
            raise RuntimeError(f"Failed to read file '{path}': {str(e)}")

    async def _arun(self, path: str, ref: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class UpdateFileTool(BaseTool):
    name: str = "update_file"
    description: str = "Update an existing file. Input: path, message, content, branch (optional), sha (optional)."
    args_schema: type[BaseModel] = UpdateFileInput

    def _run(self, path: str, message: str, content: str, branch: Optional[str] = None, sha: Optional[str] = None) -> Dict[str, Any]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            if hasattr(wrapper, "update_file"):
                return wrapper.update_file(path=path, message=message, content=content, branch=branch, sha=sha)
            
            if sha is None:
                existing = repo.get_contents(path, ref=branch)
                sha = existing.sha
            updated = repo.update_file(path, message, content, sha, branch=branch)
            return {"commit_sha": getattr(updated[0], "sha", None), "file": getattr(updated[1], "path", None)}
        except Exception as e:
            logger.error(f"Error updating file '{path}': {e}")
            raise RuntimeError(f"Failed to update file '{path}': {e}")

    async def _arun(self, path: str, message: str, content: str, branch: Optional[str] = None, sha: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class DeleteFileTool(BaseTool):
    name: str = "delete_file"
    description: str = "Delete a file. Input: path, message, branch (optional), sha (optional)."
    args_schema: type[BaseModel] = DeleteFileInput

    def _run(self, path: str, message: str, branch: Optional[str] = None, sha: Optional[str] = None) -> Dict[str, Any]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            if hasattr(wrapper, "delete_file"):
                return wrapper.delete_file(path=path, message=message, branch=branch, sha=sha)
            
            if sha is None:
                existing = repo.get_contents(path, ref=branch)
                sha = existing.sha
            deleted = repo.delete_file(path, message, sha, branch=branch)
            return {"commit_sha": getattr(deleted, "sha", None)} if deleted else {"result": "deleted"}
        except Exception as e:
            logger.error(f"Error deleting file '{path}': {e}")
            raise RuntimeError(f"Failed to delete file '{path}': {e}")

    async def _arun(self, path: str, message: str, branch: Optional[str] = None, sha: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class OverviewFilesMainBranchTool(BaseTool):
    name: str = "overview_files_main_branch"
    description: str = "List files in the repository's main (default) branch."
    args_schema: type[BaseModel] = NoInput

    def _run(self) -> List[str]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            default_branch = getattr(repo, "default_branch", None)
            return _list_files_in_repo_recursive(repo, path="", ref=default_branch)
        except Exception as e:
            logger.error(f"Error listing files in main branch: {e}")
            raise RuntimeError(f"Failed to list files in main branch: {e}")

    async def _arun(self):
        raise NotImplementedError("Async not implemented.")

class OverviewFilesBranchTool(BaseTool):
    name: str = "overview_files_current_branch"
    description: str = "List files in a specified branch. Input: branch (optional)."
    args_schema: type[BaseModel] = OverviewFilesInput

    def _run(self, branch: Optional[str] = None) -> List[str]:
        try:
            wrapper = get_wrapper()
            repo = _get_repo(wrapper)
            ref = branch or getattr(repo, "default_branch", None)
            return _list_files_in_repo_recursive(repo, path="", ref=ref)
        except Exception as e:
            logger.error(f"Error listing files in branch '{branch}': {e}")
            raise RuntimeError(f"Failed to list files in branch '{branch}': {e}")

    async def _arun(self, branch: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

def get_github_tools(repo: Optional[str] = None) -> List[BaseTool]:
    """
    Return a list of tool instances for inclusion as an agent node.
    """
    # Ensure we have a valid wrapper before creating tools
    wrapper = get_wrapper(repo=repo)
    
    return [
        # Issues
        GetIssuesTool(),
        GetIssueTool(),
        CommentOnIssueTool(github_api_wrapper=wrapper),
        SearchIssuesAndPRsTool(),
        # Pull Requests
        ListOpenPRsTool(),
        GetPullRequestTool(),
        CreatePullRequestTool(),
        ListPullRequestFilesTool(),
        # Files
        CreateFileTool(),
        ReadFileTool(),
        UpdateFileTool(),
        DeleteFileTool(),
        OverviewFilesMainBranchTool(),
        OverviewFilesBranchTool(),
    ]