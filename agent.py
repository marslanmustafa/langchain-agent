# pip install -qU langchain "langchain[anthropic]"
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

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
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in lahore"}]}
)

print(response["messages"][-1].content)