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
            return GitHubAPIWrapper(github_repository=repo)
        return github_api_wrapper
    
    # If we have a repo parameter, create a wrapper with it
    if repo:
        try:
            wrapper = GitHubAPIWrapper(github_repository=repo)
            _global_wrapper = wrapper
            logger.info(f"Created GitHubAPIWrapper for repository: {repo}")
            return wrapper
        except Exception as e:
            logger.error(f"Failed to create wrapper for repo {repo}: {e}")
    
    # Try to reuse global wrapper if it exists
    if _global_wrapper is not None:
        logger.info(f"Reusing existing wrapper for: {getattr(_global_wrapper, 'github_repository', 'unknown')}")
        return _global_wrapper
    
    # Try environment variable
    repo_env = os.getenv("GITHUB_REPOSITORY")
    if repo_env:
        try:
            wrapper = GitHubAPIWrapper(github_repository=repo_env)
            _global_wrapper = wrapper
            logger.info(f"Created GitHubAPIWrapper from env for repository: {repo_env}")
            return wrapper
        except Exception as e:
            logger.error(f"Failed to create wrapper from env: {e}")
    
    # Final fallback - create without repository (will need to be set later)
    try:
        # Set a temporary repository to avoid validation error
        wrapper = GitHubAPIWrapper(github_repository="dummy/dummy")
        _global_wrapper = wrapper
        logger.warning("Created GitHubAPIWrapper with dummy repository - must be configured before use")
        return wrapper
    except Exception as e:
        logger.error(f"Failed to create dummy wrapper: {e}")
        raise RuntimeError("Could not create GitHubAPIWrapper instance")

def configure_wrapper_repository(repo: str) -> None:
    """
    Configure the global wrapper with a specific repository.
    """
    global _global_wrapper
    if _global_wrapper is not None:
        _global_wrapper.github_repository = repo
        logger.info(f"Configured wrapper repository to: {repo}")
    else:
        logger.warning("No wrapper instance to configure")

def _get_repo(wrapper: GitHubAPIWrapper):
    """Return a PyGithub Repository instance, trying wrapper.github_repo_instance first."""
    if hasattr(wrapper, "github_repo_instance") and wrapper.github_repo_instance:
        return wrapper.github_repo_instance
    
    # Get repository name and validate it's not dummy
    repo_name = getattr(wrapper, "github_repository", None)
    if repo_name is None or repo_name == "dummy/dummy":
        raise RuntimeError("GitHub repository not configured on the wrapper.")
    
    return wrapper.github.get_repo(repo_name)

def _serialize_issue(issue) -> Dict[str, Any]:
    return {
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "state": issue.state,
        "url": getattr(issue, "html_url", None) or getattr(issue, "url", None),
        "user": getattr(issue.user, "login", None) if getattr(issue, "user", None) else None,
    }

def _serialize_pr(pr) -> Dict[str, Any]:
    return {
        "number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "state": pr.state,
        "url": getattr(pr, "html_url", None) or getattr(pr, "url", None),
        "head": getattr(pr.head, "ref", None) if getattr(pr, "head", None) else None,
        "base": getattr(pr.base, "ref", None) if getattr(pr, "base", None) else None,
        "user": getattr(pr.user, "login", None) if getattr(pr, "user", None) else None,
        "status": pr.state
    }

def _list_files_in_repo_recursive(repo, path: str = "", ref: Optional[str] = None) -> List[str]:
    """
    Recursively list file paths in repo at given ref. Uses PyGithub repo.get_contents.
    """
    files: List[str] = []
    try:
        contents = repo.get_contents(path, ref=ref)
    except Exception:
        return files
    if not isinstance(contents, list):
        # single file
        return [contents.path]
    for entry in contents:
        if entry.type == "file":
            files.append(entry.path)
        elif entry.type == "dir":
            files.extend(_list_files_in_repo_recursive(repo, entry.path, ref=ref))
    return files

# Input schemas (keeping the same as before)
class NoInput(BaseModel):
    pass

class SearchIssuesInput(BaseModel):
    query: str = Field(..., description="Search query (GitHub search syntax).")
    per_page: Optional[int] = Field(30, description="Max results to return.")

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

    def _run(self) -> List[Dict[str, Any]]:
        wrapper = get_wrapper()
        if hasattr(wrapper, "get_issues"):
            raw = wrapper.get_issues()
            try:
                return [_serialize_issue(i) for i in raw][:50]
            except Exception:
                return list(raw)
        
        repo = _get_repo(wrapper)
        issues = []
        page = 0
        per_page = 30
        while len(issues) < 50:
            try:
                page_issues = repo.get_issues(state="open", per_page=per_page, page=page)
                if not page_issues:
                    break
                issues.extend(page_issues)
                page += 1
            except Exception as e:
                logger.error(f"Error fetching issues page {page}: {str(e)}")
                break
        
        return [_serialize_issue(i) for i in issues[:50]]

    async def _arun(self):
        raise NotImplementedError("Async not implemented.")

class GetIssueTool(BaseTool):
    name: str = "get_issue"
    description: str = "Get details and comments of a specific issue. Input: issue_number (int)."
    args_schema: type[BaseModel] = GetIssueInput

    def _run(self, issue_number: int) -> Dict[str, Any]:
        wrapper = get_wrapper()
        if hasattr(wrapper, "get_issue"):
            raw = wrapper.get_issue(issue_number)
            try:
                return _serialize_issue(raw)
            except Exception:
                return raw
        
        repo = _get_repo(wrapper)
        issue = repo.get_issue(number=issue_number)
        comments = [c.body for c in issue.get_comments()]
        result = _serialize_issue(issue)
        result["comments"] = comments
        return result

    async def _arun(self, issue_number: int):
        raise NotImplementedError("Async not implemented.")

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
        wrapper = self.github_api_wrapper or get_wrapper()
        repo = _get_repo(wrapper)
        
        try:
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

class SearchIssuesAndPRsTool(BaseTool):
    name: str = "search_issues_and_prs"
    description: str = "Search issues and pull requests using GitHub search syntax. Input: query."
    args_schema: type[BaseModel] = SearchIssuesInput

    def _run(self, query: str, per_page: int = 30) -> List[Dict[str, Any]]:
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

    async def _arun(self, query: str, per_page: int = 30):
        raise NotImplementedError("Async not implemented.")

class ListOpenPRsTool(BaseTool):
    name: str = "list_open_pull_requests"
    description: str = "List open pull requests (default open)."
    args_schema: type[BaseModel] = ListPRsInput

    def _run(self, state: str = "open", per_page: int = 5) -> List[Dict[str, Any]]:
        """List open pull requests with proper pagination and limited results"""
        wrapper = get_wrapper()
        repo = _get_repo(wrapper)
        
        # Use a reasonable default for per_page (5 as requested)
        # Cap at 30 to avoid GitHub API limits
        per_page = min(per_page, 30)
        
        # Get only the first page with the requested number of PRs
        pulls = repo.get_pulls(state=state)
        
        # Convert to our format without triggering additional API calls
        pr_list = []
        try:
            # Iterate through the paginated list and limit results
            for i, pr in enumerate(pulls):
                if i >= per_page:
                    break
                pr_list.append(_serialize_pr(pr))
        except Exception as e:
            logger.error(f"Error fetching pull requests: {str(e)}")
            raise RuntimeError(f"Failed to fetch pull requests: {str(e)}")
        
        return pr_list

    async def _arun(self, state: str = "open", per_page: int = 5):
        raise NotImplementedError("Async not implemented.")

class GetPullRequestTool(BaseTool):
    name: str = "get_pull_request"
    description: str = "Get details of a pull request (number)."
    args_schema: type[BaseModel] = GetPRInput

    def _run(self, pr_number: int) -> Dict[str, Any]:
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

    async def _arun(self, pr_number: int):
        raise NotImplementedError("Async not implemented.")

class CreatePullRequestTool(BaseTool):
    name: str = "create_pull_request"
    description: str = "Create a pull request. Input: title, body, head, base."
    args_schema: type[BaseModel] = CreatePRInput

    def _run(self, title: str, head: str, base: str, body: Optional[str] = None) -> Dict[str, Any]:
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

    async def _arun(self, title: str, head: str, base: str, body: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class ListPullRequestFilesTool(BaseTool):
    name: str = "list_pull_request_files"
    description: str = "List files changed in a pull request. Input: pr_number (int)."
    args_schema: type[BaseModel] = PRFilesInput

    def _run(self, pr_number: int) -> List[Dict[str, Any]]:
        wrapper = get_wrapper()
        if hasattr(wrapper, "list_pull_request_files"):
            files = wrapper.list_pull_request_files(pr_number)
            return files
        repo = _get_repo(wrapper)
        pr = repo.get_pull(pr_number)
        files = pr.get_files()
        return [{"filename": f.filename, "status": f.status, "changes": f.changes, "raw_url": getattr(f, "raw_url", None)} for f in files]

    async def _arun(self, pr_number: int):
        raise NotImplementedError("Async not implemented.")

class CreateFileTool(BaseTool):
    name: str = "create_file"
    description: str = "Create a file in the repository. Input: path, message, content, branch (optional)."
    args_schema: type[BaseModel] = CreateFileInput

    def _run(self, path: str, message: str, content: str, branch: Optional[str] = None) -> Dict[str, Any]:
        wrapper = get_wrapper()
        if hasattr(wrapper, "create_file"):
            return wrapper.create_file(path=path, message=message, content=content, branch=branch)
        repo = _get_repo(wrapper)
        created = repo.create_file(path, message, content, branch=branch)
        return {"commit_sha": getattr(created[0], "sha", None), "file": getattr(created[1], "path", None)}

    async def _arun(self, path: str, message: str, content: str, branch: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read a file's contents. Input: path (and optionally ref for branch/commit)."
    args_schema: type[BaseModel] = ReadFileInput

    def _run(self, path: str, ref: Optional[str] = None) -> str:
        wrapper = get_wrapper()
        repo = _get_repo(wrapper)
        
        try:
            contents = repo.get_contents(path, ref=ref)
            
            if hasattr(contents, "content"):
                import base64
                return base64.b64decode(contents.content).decode('utf-8')
            return str(contents)
        except Exception as e:
            raise RuntimeError(f"Failed to read file '{path}': {str(e)}")

    async def _arun(self, path: str, ref: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class UpdateFileTool(BaseTool):
    name: str = "update_file"
    description: str = "Update an existing file. Input: path, message, content, branch (optional), sha (optional)."
    args_schema: type[BaseModel] = UpdateFileInput

    def _run(self, path: str, message: str, content: str, branch: Optional[str] = None, sha: Optional[str] = None) -> Dict[str, Any]:
        wrapper = get_wrapper()
        repo = _get_repo(wrapper)
        if hasattr(wrapper, "update_file"):
            return wrapper.update_file(path=path, message=message, content=content, branch=branch, sha=sha)
        
        if sha is None:
            existing = repo.get_contents(path, ref=branch)
            sha = existing.sha
        updated = repo.update_file(path, message, content, sha, branch=branch)
        return {"commit_sha": getattr(updated[0], "sha", None), "file": getattr(updated[1], "path", None)}

    async def _arun(self, path: str, message: str, content: str, branch: Optional[str] = None, sha: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class DeleteFileTool(BaseTool):
    name: str = "delete_file"
    description: str = "Delete a file. Input: path, message, branch (optional), sha (optional)."
    args_schema: type[BaseModel] = DeleteFileInput

    def _run(self, path: str, message: str, branch: Optional[str] = None, sha: Optional[str] = None) -> Dict[str, Any]:
        wrapper = get_wrapper()
        repo = _get_repo(wrapper)
        if hasattr(wrapper, "delete_file"):
            return wrapper.delete_file(path=path, message=message, branch=branch, sha=sha)
        if sha is None:
            existing = repo.get_contents(path, ref=branch)
            sha = existing.sha
        deleted = repo.delete_file(path, message, sha, branch=branch)
        return {"commit_sha": getattr(deleted, "sha", None)} if deleted else {"result": "deleted"}

    async def _arun(self, path: str, message: str, branch: Optional[str] = None, sha: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

class OverviewFilesMainBranchTool(BaseTool):
    name: str = "overview_files_main_branch"
    description: str = "List files in the repository's main (default) branch."
    args_schema: type[BaseModel] = NoInput

    def _run(self) -> List[str]:
        wrapper = get_wrapper()
        repo = _get_repo(wrapper)
        default_branch = getattr(repo, "default_branch", None)
        return _list_files_in_repo_recursive(repo, path="", ref=default_branch)

    async def _arun(self):
        raise NotImplementedError("Async not implemented.")

class OverviewFilesBranchTool(BaseTool):
    name: str = "overview_files_current_branch"
    description: str = "List files in a specified branch. Input: branch (optional)."
    args_schema: type[BaseModel] = OverviewFilesInput

    def _run(self, branch: Optional[str] = None) -> List[str]:
        wrapper = get_wrapper()
        repo = _get_repo(wrapper)
        ref = branch or getattr(repo, "default_branch", None)
        return _list_files_in_repo_recursive(repo, path="", ref=ref)

    async def _arun(self, branch: Optional[str] = None):
        raise NotImplementedError("Async not implemented.")

def get_github_tools(repo: Optional[str] = None) -> List[BaseTool]:
    """
    Return a list of tool instances for inclusion as an agent node.
    """
    wrapper = get_wrapper(repo=repo)
    return [
        # Issues
        GetIssuesTool(),
        GetIssueTool(),
        CommentOnIssueTool(),
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