from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

# Use tavily tools instead of custom tool
# from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query)


llm = ChatOllama(temperature=0.25, model="qwen3.5:9b")
tools = [search]
# Use tavily tools instead of custom tool
# tools = [TavilySearch()]
agent = create_agent(llm, tools)


def main() -> None:
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in San Diego today?")]}
    )
    print(response)


if __name__ == "__main__":
    main()
