# Reaserch AI Agent

from datetime import datetime

# LangChain Agent & Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware
)
from langchain_classic.agents import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# LangChain Community 
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver 

# Model

from research import create_research_agent
from langchain_ollama import ChatOllama


# --- RESEARCH TOOLS ---
ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_tool = DuckDuckGoSearchResults(
   api_wrapper=ddgs_wrapper,
   name="duckduckgo_search",
   description="Search the live web for current news, events, and general information."
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
   api_wrapper=wiki_wrapper,
   name="wikipedia",
   description="Search Wikipedia for historical facts, biographies, and high-level summaries."
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(
   api_wrapper=arxiv_wrapper,
   name="arxiv",
   description="Search ArXiv for scientific papers, LaTeX-based research, and technical depth."
)

@tool    #optional decorator to mark this function as a tool for the agent
def get_current_datetime():
    """Returns the current date and time as a string. Use this to orient your searches."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



tools = [ddgs_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_RESEARCH_PROMPT = """You are a Meticulous Research Agent specialized in factual accuracy and deep synthesis.

### OPERATIONAL GUIDELINES:
1. **Orientation:** Always call `get_current_datetime` first if the query involves 'today', 'latest', or 'recent' to establish a temporal baseline.
2. **Triangulation & Tool Selection:**
    - Use **Wikipedia** for foundational facts, history, and definitions.
    - Use **DuckDuckGo** for real-time news, recent events, and specific URLs.
    - Use **ArXiv** for academic papers and deep technical/scientific rigor.
    - *Example:* If asked about a new AI model, check Wikipedia for the creator's history, ArXiv for the architecture, and DuckDuckGo for the release date.
3. **Source Integrity:** You must never provide information without a source. If a tool returns a URL or Paper ID, you must include it.
4. **Formatting:** Use Markdown headers (##) for clarity and bullet points for data synthesis.

### CITATION STYLE:
- **In-text:** Use [Source Name/Link] immediately after the claim.
- **References Section:** Always end your response with a '### References' section listing all unique sources used.

### CONSTRAINT:
If you cannot find the answer after 3 tool attempts, admit the limitation rather than hallucinating. Do not repeat the same search query across different tools if it fails the first time."""


def create_reasearch_agent():
    llm = ChatOllama(
        model="minimax-m2.5:cloud",
        temperature=0
        )
    
    memory = MemorySaver()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        middleware="",
        name="research_agent",
        checkpointer=memory
    )
    
    return agent

def banner():
    print("="*50)
    print("Welcome to the Research Agent!")
    print("Ask me anything about current events, historical facts, or scientific research.")
    print("Type 'clear' to reset context, 'exit' to quit.")
    print("="*50)

def stream_response(agent, query: str, config: dict):
    stream = agent.stream(query, config=config, stream_mode="values")
    for chunk in stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="values"):
    # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            # if isinstance(latest_message, HumanMessage):
                # print(f"User: {latest_message.content}")
            if isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


def main():
    banner()
    agent = create_reasearch_agent()
    config = {"configurable": {"thread_id": "research-session_001"}}
    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt as err:
            print("\n Goodbye!")
            break
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("Exiting chatbot. Goodbye!")
            break
        
        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error during agent execution: {err}")
            continue
main()