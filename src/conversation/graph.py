from typing import TypedDict, Sequence, Dict, Any
from langgraph.graph import Graph, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
import os
from functools import lru_cache

# Global LLM instance
@lru_cache(maxsize=1)
def get_llm():
    """Get a shared LLM instance."""
    print("Creating new LLM instance")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return ChatOpenAI(
        model="gpt-4", 
        temperature=0, 
        api_key=api_key,
        max_retries=3,
        request_timeout=30
    )

class StandupState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage | FunctionMessage]
    user_info: Dict[str, Any]
    current_draft: Dict[str, Any]
    activities: Dict[str, Any]
    next_step: str

def initialize_state(state: StandupState) -> StandupState:
    """Initialize the conversation state."""
    print("Executing initialize_state")
    state["next_step"] = "generate_draft"
    return state

def generate_draft(state: StandupState) -> StandupState:
    """Generate initial draft based on activities."""
    print("Executing generate_draft")
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant helping to generate standup updates. Create a draft based on the user's message."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Generate a draft standup update based on my previous message.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "messages": state["messages"]
    })
    
    state["messages"].append(AIMessage(content=response.content))
    state["next_step"] = "end"
    return state

def create_standup_graph() -> Graph:
    """Create a simple two-step standup graph."""
    workflow = StateGraph(StandupState)
    
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("generate_draft", generate_draft)
    
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "generate_draft")
    workflow.add_edge("generate_draft", END)
    
    return workflow.compile()