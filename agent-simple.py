# pip install -qU langchain "langchain[anthropic]"
import os
from dotenv import load_dotenv

# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langchain.agents import create_agent
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
temperature = os.getenv("TEMPERATURE")

# llm = ChatGroq(
#       model=model_name,
#       temperature=float(temperature),
#       api_key=api_key,
#   )

#LLM
llm = ChatOllama(
      model=model_name,
      temperature=float(temperature),
  )

#Tool
def get_weather(city: str) -> str:
    """Get information for a given city."""
    return f"It's always sunny in {city}!"

web_search = DuckDuckGoSearchRun()
# wiki_search = WikipediaQueryRun(
#     name="wikipedia",  #for the AI Help to know what is this class
#     description="Useful for when you need to answer questions about general knowledge or current events.",
# )
# arxiv_search = ArxivQueryRun()

agent = create_agent(
    model=llm,
    tools=[get_weather, web_search],
    system_prompt="You are a helpful assistant that can anser from the web",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "who is marslanmustafa or marslanmustafa.com"}]}
)

print(response["messages"][-1].content)
