from typing import TypedDict, Annotated, Sequence, Dict, Any
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
import json
import os

# Initialize OpenAI client with API key
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)

# State definition
class StandupState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage | FunctionMessage]
    user_info: Dict[str, Any]
    current_draft: Dict[str, Any]
    activities: Dict[str, Any]
    next_step: str

# Node functions
def initialize_state(state: StandupState) -> StandupState:
    """Initialize the conversation state."""
    state["next_step"] = "fetch_activities"
    return state

def fetch_activities(state: StandupState) -> StandupState:
    """Fetch GitHub and Linear activities."""
    # This would be replaced with actual API calls
    activities = {
        "accomplishments": state["activities"].get("accomplishments", []),
        "plans": state["activities"].get("plans", []),
        "blockers": state["activities"].get("blockers", [])
    }
    state["activities"] = activities
    state["next_step"] = "generate_draft"
    return state

def generate_draft(state: StandupState) -> StandupState:
    """Generate initial draft based on activities."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant helping to generate standup updates. 
        Based on the user's activities and any previous messages, create a well-structured draft.
        Format the response as JSON with keys: accomplishments, plans, blockers."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Activities: {activities}
        Generate a draft standup update.""")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "messages": state["messages"],
        "activities": json.dumps(state["activities"], indent=2)
    })
    
    try:
        draft = json.loads(response.content)
    except:
        draft = {
            "accomplishments": state["activities"].get("accomplishments", []),
            "plans": state["activities"].get("plans", []),
            "blockers": state["activities"].get("blockers", [])
        }
    
    state["current_draft"] = draft
    state["next_step"] = "analyze_draft"
    return state

def analyze_draft(state: StandupState) -> StandupState:
    """Analyze draft for completeness and clarity."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant analyzing standup updates.
        Check if the draft needs clarification or has missing information.
        Return JSON with keys: needs_clarification (boolean), questions (list of strings)."""),
        ("human", """Draft: {draft}
        Analyze this draft for completeness and clarity.""")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "draft": json.dumps(state["current_draft"], indent=2)
    })
    
    try:
        analysis = json.loads(response.content)
        if analysis["needs_clarification"]:
            state["next_step"] = "ask_followup"
            state["messages"].append(
                AIMessage(content="\n".join(analysis["questions"]))
            )
        else:
            state["next_step"] = "finalize"
    except:
        state["next_step"] = "finalize"
    
    return state

def ask_followup(state: StandupState) -> StandupState:
    """Process user's response to follow-up questions."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant updating a standup draft based on follow-up responses.
        Update the draft with the new information and return as JSON."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Current draft: {draft}
        Update the draft based on the conversation history.""")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "messages": state["messages"],
        "draft": json.dumps(state["current_draft"], indent=2)
    })
    
    try:
        updated_draft = json.loads(response.content)
        state["current_draft"] = updated_draft
    except:
        pass
    
    state["next_step"] = "analyze_draft"
    return state

def finalize(state: StandupState) -> StandupState:
    """Finalize the standup update."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant formatting the final standup update.
        Create a well-formatted message suitable for posting in a team channel."""),
        ("human", """Draft: {draft}
        User: {user}
        Create the final formatted message.""")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "draft": json.dumps(state["current_draft"], indent=2),
        "user": state["user_info"].get("name", "User")
    })
    
    state["messages"].append(AIMessage(content=response.content))
    state["next_step"] = "end"
    return state

# Graph definition
def create_standup_graph() -> Graph:
    """Create the standup conversation workflow graph."""
    workflow = StateGraph(StandupState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("fetch_activities", fetch_activities)
    workflow.add_node("generate_draft", generate_draft)
    workflow.add_node("analyze_draft", analyze_draft)
    workflow.add_node("ask_followup", ask_followup)
    workflow.add_node("finalize", finalize)
    
    # Add edges
    workflow.add_edge("initialize", "fetch_activities")
    workflow.add_edge("fetch_activities", "generate_draft")
    workflow.add_edge("generate_draft", "analyze_draft")
    workflow.add_edge("analyze_draft", "ask_followup")
    workflow.add_edge("analyze_draft", "finalize")
    workflow.add_edge("ask_followup", "analyze_draft")
    
    # Set conditional edges
    workflow.add_conditional_edges(
        "analyze_draft",
        lambda x: x["next_step"],
        {
            "ask_followup": "ask_followup",
            "finalize": "finalize"
        }
    )
    
    workflow.set_entry_point("initialize")
    return workflow.compile()
