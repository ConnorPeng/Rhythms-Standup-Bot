from typing import TypedDict, Sequence, Dict, Any
from langgraph.graph import Graph, StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
import os
from functools import lru_cache
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global LLM instance
@lru_cache(maxsize=1)
def get_llm():
    """Get a shared LLM instance."""
    print("Creating new LLM instance")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        max_tokens=1024,
        max_retries=2,
        anthropic_api_key=api_key,
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

import threading

# Create a global lock
llm_lock = threading.Lock()

def generate_draft(state: StandupState) -> StandupState:
    """Generate initial draft based on activities."""
    logger.info("Executing generate_draft")
    try:
        # Ensure that only one thread can execute the LLM call at a time
        with llm_lock:
            llm = get_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant helping to generate standup updates."),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "Generate a draft standup update based on my previous message.")
            ])
            logger.info("prompt: %s", prompt)
            chain = prompt | llm
            logger.info("Invoking LLM, chain: %s", chain)
            response = chain.invoke({"messages": state["messages"]})
            state["messages"].append(AIMessage(content=response.content))
        state["next_step"] = "end"
    except Exception as e:
        logger.error(f"Unexpected Error: {e}", exc_info=True)
        state["messages"].append(AIMessage(content="An unexpected error occurred. Please try again later."))
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