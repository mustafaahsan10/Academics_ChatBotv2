"""
Agent SDK Template using LangGraph and OpenRouter

This template provides a framework for building agents similar to OpenAI's Agent SDK
but using LangGraph for the agent architecture and OpenRouter for model access.
"""

import os
import json
import requests
from typing import Dict, List, Any, Callable, TypedDict, Optional, Union, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import inspect
from dataclasses import dataclass, field

# --------------------------------
# Configuration and Settings
# --------------------------------
class AgentConfig(BaseModel):
    """Configuration for the agent."""
    api_key: str = Field(..., description="OpenRouter API key")
    model: str = Field(default="openai/gpt-4-turbo", description="Model to use for the agent")
    temperature: float = Field(default=0.7, description="Temperature for model generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    tools: List[Any] = Field(default_factory=list, description="List of tools available to the agent")
    instructions: str = Field(default="", description="System instructions for the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# --------------------------------
# Tool Definition
# --------------------------------
class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str = Field(..., description="Name of the parameter")
    description: str = Field(..., description="Description of the parameter")
    type: str = Field(..., description="Type of the parameter (e.g., string, integer)")
    required: bool = Field(default=False, description="Whether the parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value for the parameter")

class Tool(BaseModel):
    """Definition of a tool that the agent can use."""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Parameters for the tool")
    function: Callable = Field(..., description="Function to execute when the tool is called")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary format for the LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description,
                        } for param in self.parameters
                    },
                    "required": [param.name for param in self.parameters if param.required]
                }
            }
        }

# --------------------------------
# Message Types
# --------------------------------
class Message(BaseModel):
    """A message in the conversation."""
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(default=None, description="Name of the sender (for tool messages)")
    tool_call_id: Optional[str] = Field(default=None, description="ID of the tool call (for tool messages)")

# --------------------------------
# State Management
# --------------------------------
class AgentState(TypedDict):
    """State of the agent."""
    messages: List[Dict[str, Any]]
    next_steps: List[str]
    current_tool: Optional[str]
    tool_results: Dict[str, Any]
    thinking: str
    final_response: Optional[str]

# --------------------------------
# OpenRouter Integration
# --------------------------------
def call_openrouter(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Call the OpenRouter API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-app-url.com",  # Replace with your app URL
        "X-Title": "Agent SDK Template",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
        
    if tools:
        payload["tools"] = tools
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()

# --------------------------------
# Core Agent Logic
# --------------------------------
def route_step(state: AgentState) -> Literal["execute_tool", "process_results", END]:
    """Determine the next step in the agent workflow."""
    if not state["next_steps"]:
        return END
    
    next_step = state["next_steps"].pop(0)
    return next_step

def analyze_user_input(state: AgentState, config: AgentConfig) -> AgentState:
    """Analyze user input and determine next steps."""
    # Make initial call to LLM to analyze the request
    tools_dict = [tool.to_dict() for tool in config.tools] if config.tools else None
    
    response = call_openrouter(
        messages=state["messages"],
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        tools=tools_dict
    )
    
    assistant_message = response["choices"][0]["message"]
    state["messages"].append(assistant_message)
    
    # Check if the assistant wants to use a tool
    if "tool_calls" in assistant_message:
        state["next_steps"].append("execute_tool")
        state["current_tool"] = assistant_message["tool_calls"][0]["function"]["name"]
    else:
        state["final_response"] = assistant_message["content"]
        state["next_steps"] = []  # End the conversation
    
    return state

def execute_tool(state: AgentState, config: AgentConfig) -> AgentState:
    """Execute the specified tool and update the state."""
    tool_calls = state["messages"][-1].get("tool_calls", [])
    tool_results = {}
    
    for tool_call in tool_calls:
        function_call = tool_call["function"]
        tool_name = function_call["name"]
        arguments = json.loads(function_call["arguments"])
        
        # Find the matching tool
        tool = next((t for t in config.tools if t.name == tool_name), None)
        if tool:
            try:
                result = tool.function(**arguments)
                tool_results[tool_call["id"]] = result
                
                # Add tool result to messages
                state["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": str(result)
                })
            except Exception as e:
                tool_results[tool_call["id"]] = f"Error: {str(e)}"
                state["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": f"Error: {str(e)}"
                })
    
    state["tool_results"].update(tool_results)
    state["next_steps"].append("process_results")
    return state

def process_results(state: AgentState, config: AgentConfig) -> AgentState:
    """Process tool results and determine next actions."""
    # Make a call to the LLM to process tool results
    tools_dict = [tool.to_dict() for tool in config.tools] if config.tools else None
    
    response = call_openrouter(
        messages=state["messages"],
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        tools=tools_dict
    )
    
    assistant_message = response["choices"][0]["message"]
    state["messages"].append(assistant_message)
    
    # Check if the assistant wants to use another tool
    if "tool_calls" in assistant_message:
        state["next_steps"].append("execute_tool")
        state["current_tool"] = assistant_message["tool_calls"][0]["function"]["name"]
    else:
        state["final_response"] = assistant_message["content"]
        state["next_steps"] = []  # End the conversation
    
    return state

# --------------------------------
# Agent Creation and Execution
# --------------------------------
class Agent:
    """An agent that can process requests and use tools."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with the given configuration."""
        self.config = config
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Any:
        """Build the agent workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_user_input", lambda state: analyze_user_input(state, self.config))
        workflow.add_node("execute_tool", lambda state: execute_tool(state, self.config))
        workflow.add_node("process_results", lambda state: process_results(state, self.config))
        
        # Add direct edge from START to analyze_user_input
        workflow.add_edge(START, "analyze_user_input")
        
        # Define routing functions for conditional edges
        def route_from_analyze(state: AgentState) -> str:
            """Route from analyze_user_input node"""
            if "tool_calls" in state["messages"][-1]:
                return "execute_tool"
            return END
        
        def route_from_execute(state: AgentState) -> str:
            """Route from execute_tool node"""
            return "process_results"
        
        def route_from_process(state: AgentState) -> str:
            """Route from process_results node"""
            if "tool_calls" in state["messages"][-1]:
                return "execute_tool"
            return END
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_user_input",
            route_from_analyze,
            {
                "execute_tool": "execute_tool",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tool",
            route_from_execute,
            {
                "process_results": "process_results"
            }
        )
        
        workflow.add_conditional_edges(
            "process_results",
            route_from_process,
            {
                "execute_tool": "execute_tool",
                END: END
            }
        )
        
        # Compile the graph
        return workflow.compile()
    
    def run(self, user_input: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run the agent with the given user input."""
        # Initialize messages
        if conversation_history is None:
            messages = []
            
            # Add system message if instructions are provided
            if self.config.instructions:
                messages.append({
                    "role": "system",
                    "content": self.config.instructions
                })
        else:
            messages = conversation_history.copy()
        
        # Add user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Initialize state
        initial_state: AgentState = {
            "messages": messages,
            "next_steps": ["analyze_user_input"],
            "current_tool": None,
            "tool_results": {},
            "thinking": "",
            "final_response": None
        }
        
        # Execute the agent
        final_state = self.graph.invoke(initial_state)
        
        return {
            "response": final_state["final_response"],
            "messages": final_state["messages"],
            "tool_results": final_state["tool_results"]
        }

# --------------------------------
# Example Usage
# --------------------------------
def example_weather_tool(location: str) -> str:
    """Example tool that gets the weather for a location."""
    # In a real implementation, this would call a weather API
    return f"The weather in {location} is currently sunny and 75°F."

# Example of how to create and use an agent
if __name__ == "__main__":
    # Define tools
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters=[
            ToolParameter(
                name="location",
                description="The city and state or country",
                type="string",
                required=True
            )
        ],
        function=example_weather_tool
    )
    
    # Create agent configuration
    config = AgentConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model="anthropic/claude-3-sonnet",
        temperature=0.7,
        tools=[weather_tool],
        instructions="""You are a helpful assistant that can answer questions and use tools 
        to find information. Always be concise and accurate in your responses."""
    )
    
    # Create the agent
    agent = Agent(config)
    
    # Run the agent with a user query
    result = agent.run("What's the weather like in San Francisco?")
    print("Agent response:", result["response"]) 