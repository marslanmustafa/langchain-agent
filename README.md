Based on my research, here are the code snippets for streaming agent results in LangChain:

## Streaming Agent Progress (stream_mode="updates")

```python
# Stream agent progress - emits an event after each agent step
for step in agent.stream({"messages": [("user", "Your message here")]}, stream_mode="updates"):
    for _, update in step.items():
        if update:
            print(update)
```

## Streaming Agent with Async

```python
# Async streaming
async for chunk in agent.astream(
    {"messages": [("user", "Your message here")]}, 
    stream_mode="updates"
):
    print(chunk)
```

## Streaming with stream_events (for token-level streaming)

```python
# Stream events including LLM tokens
async for chunk in agent.stream_events("Search for the latest Python news"):
    print(chunk)
```

## Streaming with LangGraph (create_react_agent)

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Create agent
agent = create_react_agent(model, tools)

# Stream with messages mode
async for mode, chunk in agent.astream(
    {"messages": [{"role": "user", "content": "Your question"}]},
    stream_mode="messages"
):
    if hasattr(chunk, 'content'):
        print(chunk.content, end="", flush=True)
```

## Key Points

| Stream Mode | Description |
|-------------|-------------|
| `"updates"` | Emits state updates after each agent step |
| `"messages"` | Streams LLM tokens as they're generated |
| `["updates", "messages"]` | Multiple modes at once |

The `stream_mode="updates"` is the most common for seeing agent progress (tool calls, intermediate steps), while `"messages"` gives you token-by-token streaming like standard LLM streaming.

Would you like me to elaborate on any specific use case?