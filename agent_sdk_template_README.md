# Agent SDK Template

This template provides a framework for building agents similar to OpenAI's Agent SDK but using LangGraph for the agent architecture and OpenRouter for model access.

## Overview

The Agent SDK template offers:

- A structured approach to building AI agents with tool-using capabilities
- LangGraph-based workflow management for complex agent behaviors
- OpenRouter integration for access to a variety of LLM providers
- Tool definition and execution framework
- Conversation state management

## Requirements

```
pydantic
requests
langgraph>=0.0.27
```

## Getting Started

1. First, install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

3. Import the template and define your tools:

```python
from agent_sdk_template import Agent, AgentConfig, Tool, ToolParameter

# Define a custom tool
def my_custom_tool(param1, param2):
    # Tool implementation
    return f"Processed {param1} and {param2}"

tool = Tool(
    name="custom_tool",
    description="A custom tool that processes two parameters",
    parameters=[
        ToolParameter(
            name="param1",
            description="First parameter",
            type="string",
            required=True
        ),
        ToolParameter(
            name="param2",
            description="Second parameter",
            type="string",
            required=True
        )
    ],
    function=my_custom_tool
)

# Configure and create your agent
config = AgentConfig(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    model="anthropic/claude-3-sonnet",  # Or any other model supported by OpenRouter
    tools=[tool],
    instructions="You are a helpful assistant that can use tools to accomplish tasks."
)

agent = Agent(config)

# Run the agent with a user query
result = agent.run("I need to process 'data1' and 'data2'")
print(result["response"])
```

## Key Components

### AgentConfig

Configures the agent with:
- API key for OpenRouter
- Model to use
- Tools available to the agent
- System instructions

### Tool and ToolParameter

Define tools that the agent can use, including:
- Name and description
- Parameters with types and requirements
- Function to execute when called

### Agent

The main class that:
- Builds the LangGraph workflow
- Processes user input
- Manages the conversation state
- Executes tools
- Returns responses

## Advanced Usage

### Conversation History

You can maintain conversation history between runs:

```python
conversation = []
result1 = agent.run("First question", conversation_history=conversation)
conversation = result1["messages"]
result2 = agent.run("Follow-up question", conversation_history=conversation)
```

### Custom Workflow

You can extend the Agent class to customize the workflow:

```python
class CustomAgent(Agent):
    def _build_graph(self):
        # Custom graph building logic
        workflow = super()._build_graph()
        # Add custom nodes and edges
        return workflow
```

## Comparison with OpenAI Agent SDK

This template provides similar functionality to OpenAI's Agent SDK:
- Tool definition and execution
- Conversation management
- Multi-step reasoning

The key differences are:
- Uses LangGraph for workflow management instead of OpenAI's proprietary system
- Supports multiple LLM providers through OpenRouter
- More customizable workflow structure
- Open-source and extensible

## License

This template is provided as open-source software. 