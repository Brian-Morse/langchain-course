from dotenv import load_dotenv
from langchain.agents import create_agent

# from langchain.tools import tool

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# from tavily import TavilyClient

from typing import List
from pydantic import BaseModel, Field

# Use tavily tools instead of custom tool
from langchain_tavily import TavilySearch

load_dotenv()

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query)


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer",
    )


# NOTE: Structured output is not supported by Ollama
# So we need to use the response_format parameter to create the agent
# and then use the structured_output method to get the structured output
# from the agent's response

llm = ChatOllama(temperature=0.25, model="qwen3.5:9b")
# tools = [search]
# Use tavily tools instead of custom tool
tools = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)


def main() -> None:
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for a machine learning engineer in the san diego area on LinkedIn and provide the links and information about the job postings"
                )
            ]
        }
    )
    print(response)


if __name__ == "__main__":
    main()
