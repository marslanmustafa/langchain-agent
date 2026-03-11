import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import tiktoken
from langchain_community.tools import ( DuckDuckGoSearchRun )
from langchain.agents import create_agent

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")
temperature = os.getenv("TEMPERATURE")
max_turns = os.getenv("MAX_TURNS")
max_tokens = os.getenv("MAX_TOKENS")

llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "groq":
    llm = ChatGroq(
        model=model_name,
        temperature=float(temperature),
        api_key=api_key,
    )
else:
    llm = ChatOllama(
        model=model_name,
        temperature=float(temperature),
    )

def get_weather(city: str) -> str:
    """Get information for a given city."""
    return f"It's always sunny in {city}!"

web_search = DuckDuckGoSearchRun()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in AI, Give concise, accurate answers."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

chat_history = []


agent = create_agent(
    model=llm,
    tools=[get_weather, web_search],
    system_prompt="You are a helpful assistant that can anser from the web",
)

chain = prompt | agent | StrOutputParser() 


def count_tokens(text):
    """Counts tokens using tiktoken. Defaults to cl100k_base encoding."""
    model_name = "cl100k_base"
    encoding = tiktoken.get_encoding(model_name)      
    token_count = len(encoding.encode(text))
    print(f"[Token Count Log] Calculated {token_count} tokens for the given text.")
    return token_count


def chat(question):
    history_text = "\n".join([msg.content for msg in chat_history])
    current_tokens = count_tokens(history_text + question)
    
    if current_tokens >= max_tokens:
        return (
            f"Your Context Window is full. (Used ~{current_tokens}/{max_tokens} tokens).\n"
            "The AI may not follow your prev threads.\n"
            "Please type 'clear' for a new chat."
        )
    
    stream = chain.stream({"question": question, "chat_history": chat_history})
    print("AI:", end=" ")
    response = ""
    for chunk in stream:
        print(chunk, end="", flush=True) 
        response += chunk
        
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    
    # Update token count
    new_history_text = "\n".join([msg.content for msg in chat_history])
    new_tokens = count_tokens(new_history_text)
    remaining = max_tokens - new_tokens
    
    print(f"\n\n[Warning: ~{remaining} token(s) left]")
    return ""  # We already streamed the response


def main():
    print("Langchain Agent bot Ready! (type 'exit' to quit) type 'clear' to reset context")
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Exiting chatbot. Goodbye!")
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared. Starting fresh!")
            continue
        print(f"AI: {chat(user_input)}")
        
main()