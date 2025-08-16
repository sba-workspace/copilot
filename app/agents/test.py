from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()


github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)

tools = toolkit.get_tools()

# for tool in tools:
#     print(tool.name)


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

from langgraph.prebuilt import create_react_agent

tools = [tool for tool in toolkit.get_tools() if tool.name == "Comment on Issue"]
assert len(tools) == 1
tools[0].name = "Comment_issues"

agent_executor = create_react_agent(llm, tools)

example_query = "Comment on issue #505 with 'This is a helpful comment about the bug"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()