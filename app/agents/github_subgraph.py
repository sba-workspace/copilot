
import os
from typing import List, Dict, Any
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from app.tools.github_tools import get_github_tools, GITHUB_TOOL_NAMES




class GitHubAgent:
    """
    An intelligent GitHub agent that can understand natural language queries
    and automatically select and execute the appropriate GitHub tools.
    """
    
    def __init__(self, github_token: str = None, openai_api_key: str = None):
        """
        Initialize the GitHub Agent.
        
        Args:
            github_token: GitHub personal access token
            openai_api_key: OpenAI API key for the LLM
        """
        # Set up environment variables if provided
        if github_token:
            os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_token
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
        # Initialize LLM
        self.llm = init_chat_model(
            model="gemini-2.5-flash", 
            temperature=0,
            model_kwargs={"seed": 42}
        )
        
        # Get GitHub tools
        self.tools = get_github_tools()
        
        # Create agent prompt
        self.prompt = self._create_agent_prompt()
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the agent prompt with GitHub-specific instructions."""
        
        system_prompt = """You are a helpful GitHub assistant that can help users interact with GitHub repositories, issues, pull requests, and files.

You have access to the following GitHub tools:
- Repository tools: Get repository information
- Issue tools: Search, get, list, and comment on issues
- Pull Request tools: List, get, create PRs and list PR files  
- File tools: Create, read, update, delete, and list files in repositories

When a user asks about GitHub-related tasks, you should:

1. **Identify the repository**: If not explicitly mentioned, ask the user to specify the repository in "owner/repo" format.

2. **Understand the task**: Determine what the user wants to do:
   - Get specific issue/PR by number → use get_github_issue or get_github_pull_request
   - Search for issues/PRs → use search_github_issues
   - List issues/PRs → use get_github_issues or list_github_pull_requests
   - Work with files → use appropriate file tools
   - Get repo info → use get_github_repo

3. **Extract parameters**: From the user's query, extract:
   - Repository name (owner/repo format)
   - Issue/PR numbers
   - File paths
   - Search terms
   - Branch names (default to "main" if not specified)

4. **Execute and respond**: Use the appropriate tool and provide a clear, helpful response.

Examples of query interpretation:
- "What is GitHub issue number 3254" → Need repo name, then use get_github_issue
- "Show me open issues in facebook/react" → use get_github_issues with repo "facebook/react"
- "Search for bugs in tensorflow" → use search_github_issues with query "repo:tensorflow/tensorflow bugs"
- "Read the README file from microsoft/vscode" → use read_github_file with path "README.md"

Always be helpful and ask for clarification if needed information is missing.
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def query(self, user_input: str, repo_context: str = None) -> str:
        """
        Process a natural language query about GitHub.
        
        Args:
            user_input: User's natural language query
            repo_context: Optional repository context to avoid asking repeatedly
            
        Returns:
            Agent's response
        """
        # Add repo context to the input if provided
        if repo_context:
            enhanced_input = f"Repository context: {repo_context}\nUser query: {user_input}"
        else:
            enhanced_input = user_input
            
        try:
            result = self.agent_executor.invoke({"input": enhanced_input})
            return result["output"]
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def aquery(self, user_input: str, repo_context: str = None) -> str:
        """Async version of query method."""
        if repo_context:
            enhanced_input = f"Repository context: {repo_context}\nUser query: {user_input}"
        else:
            enhanced_input = user_input
            
        try:
            result = await self.agent_executor.ainvoke({"input": enhanced_input})
            return result["output"]
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"


# Example usage and testing
def main():
    """Example usage of the GitHub Agent."""
    
    # Initialize agent (you'll need to set your tokens)
    agent = GitHubAgent(
        # github_token="your_github_token_here",
        # openai_api_key="your_openai_key_here"
    )
    
    # Example queries
    example_queries = [
        "What is GitHub issue number 3254 in microsoft/vscode?",
        "Show me open issues in facebook/react",
        "Search for memory leak issues in tensorflow/tensorflow",
        "Read the package.json file from facebook/react",
        "List all pull requests in microsoft/TypeScript",
        "What's in the main branch of google/go-github?",
        "Create a file called test.py with print('hello') in my-repo",
    ]
    
    print("GitHub Agent Examples:")
    print("=" * 50)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        # For demonstration, we'll just show what would happen
        # In real usage, uncomment the line below:
        # response = agent.query(query)
        # print(f"Response: {response}")
        
        print("Agent would analyze this query and:")
        
        if "issue number" in query.lower():
            print("→ Use get_github_issue tool")
            print("→ Extract repo name and issue number")
            
        elif "open issues" in query.lower():
            print("→ Use get_github_issues tool")
            print("→ Set state parameter to 'open'")
            
        elif "search for" in query.lower():
            print("→ Use search_github_issues tool")
            print("→ Construct appropriate search query")
            
        elif "read" in query.lower() and "file" in query.lower():
            print("→ Use read_github_file tool")
            print("→ Extract file path and repo name")
            
        elif "pull requests" in query.lower():
            print("→ Use list_github_pull_requests tool")
            
        elif "main branch" in query.lower():
            print("→ Use list_github_files tool")
            print("→ Set branch to 'main'")
            
        elif "create a file" in query.lower():
            print("→ Use create_github_file tool")
            print("→ Extract file path, content, and repo name")


# LangGraph Integration Example
def create_github_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function that uses the GitHub agent.
    
    Args:
        state: LangGraph state dictionary
        
    Returns:
        Updated state with agent response
    """
    # Initialize agent
    agent = GitHubAgent()
    
    # Get user query from state
    user_query = state.get("query", "")
    repo_context = state.get("repo_context", None)
    
    # Process query
    response = agent.query(user_query, repo_context)
    
    # Update state
    state["github_response"] = response
    state["last_tool_used"] = "github_agent"
    
    return state


# Alternative: Simple function-based approach
def smart_github_query(query: str, github_token: str = None) -> str:
    """
    Simple function that processes GitHub queries intelligently.
    
    Args:
        query: Natural language query about GitHub
        github_token: GitHub personal access token
        
    Returns:
        Response from the appropriate GitHub tool
    """
    agent = GitHubAgent(github_token=github_token)
    return agent.query(query)


if __name__ == "__main__":
    main()