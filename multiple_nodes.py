"""
A chatbot application demonstrating a multi-node LangGraph setup.

This script defines a stateful graph that classifies user messages
and routes them to different specialized agents (therapist or logical)
based on the classification. It uses LangChain for language model
interaction and Pydantic for data validation. The conversation state
is managed by LangGraph, allowing for complex conversational flows.
The script provides a command-line interface for user interaction.
"""

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load Environment Variables
load_dotenv()

# Initialize Chat Model
llm: BaseChatModel = init_chat_model(model="gemma3:4b", model_provider="ollama")


# Message classifier structured output
class MessageClassifier(BaseModel):
    """
    Defines the structured output for message classification.

    Attributes:
        message_type: Classifies the message as either "emotional" or "logical".
    """

    message_type: Literal["emotional", "logical"] = Field(
        default=...,
        description="Classify if the messages require an emotional(therapist) or logical response.",
    )


class State(TypedDict):
    """
    Represents the state of the conversation graph.

    Attributes:
        messages: A list of messages, managed by `add_messages` to append new messages.
        message_type: The classified type of the latest message ("emotional", "logical", or None).
    """

    messages: Annotated[list, add_messages]
    message_type: str | None


# Message Classifier Agent
def message_classify_agent(state: State) -> dict[str, Any | None]:
    """
    Classifies the latest user message as either 'emotional' or 'logical'.

    Args:
        state: The current state of the graph, containing the message history.

    Returns:
        A dictionary updating the 'message_type' in the state.
    """
    latest_message = state["messages"][-1]
    message_classifier_llm = llm.with_structured_output(schema=MessageClassifier)
    response: dict[str, Any] | BaseModel = message_classifier_llm.invoke(
        input=[
            {
                "role": "system",
                "content": """Classify the user message as either:
                - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems.
                - 'logical': if it asks for facts, information, logical analysis, or practical solutions.
                """,
            },
            {"role": "user", "content": latest_message.content},
        ]
    )

    # Handle both dict and BaseModel response types
    if isinstance(response, dict):
        message_type = response.get("message_type")
    else:
        message_type = getattr(response, "message_type", None)
    return {"message_type": message_type}


# Rout messages type
def router(state: State) -> dict[str, str]:
    """
    Determines the next node in the graph based on the classified message type.

    Args:
        state: The current state of the graph, containing the classified 'message_type'.

    Returns:
        A dictionary indicating the next node to transition to ('therapist' or 'logical').
    """
    message_type: str | None = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}


# Therapist Agent
def therapist_agent(state: State) -> dict[str, list[dict[str, Any]]]:
    """
    Handles messages classified as 'emotional' by providing a compassionate, therapeutic response.

    Args:
        state: The current state of the graph, containing the message history.

    Returns:
        A dictionary with the therapist LLM's response appended to the messages list.
    """
    latest_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked.""",
        },
        {"role": "user", "content": latest_message.content},
    ]
    response: BaseMessage = llm.invoke(input=messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


# Logical Agent
def logical_agent(state: State) -> dict[str, list[dict[str, Any]]]:
    """
    Handles messages classified as 'logical' by providing a factual, information-based response.

    Args:
        state: The current state of the graph, containing the message history.

    Returns:
        A dictionary with the logical LLM's response appended to the messages list.
    """
    latest_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.""",
        },
        {"role": "user", "content": latest_message.content},
    ]
    response: BaseMessage = llm.invoke(input=messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


# Initialize State Graph
state_graph = StateGraph(state_schema=State)
state_graph.add_node(node="classifier", action=message_classify_agent)
state_graph.add_node(node="router", action=router)
state_graph.add_node(node="therapist", action=therapist_agent)
state_graph.add_node(node="logical", action=logical_agent)

state_graph.add_edge(start_key=START, end_key="classifier")
state_graph.add_edge(start_key="classifier", end_key="router")

state_graph.add_conditional_edges(
    source="router",
    path=lambda state: state.get("next"),
    path_map={
        "therapist": "therapist",
        "logical": "logical",
    },
)

state_graph.add_edge(start_key="therapist", end_key=END)
state_graph.add_edge(start_key="logical", end_key=END)

# Compile state graph
graph: CompiledStateGraph = state_graph.compile()


# Main entrypoint
def main() -> None:
    """
    Runs the main interaction loop for the multi-agent chatbot.

    Initializes the conversation state and continuously prompts the user
    for input. Each input is processed by the LangGraph, which classifies
    the message and routes it to the appropriate agent (therapist or logical).
    The agent's response is then printed to the console.
    The loop continues until the user types 'x' to exit.
    """
    state: dict[str, Any] = {"messages": [], "message_type": None}
    while True:
        user_input: str = input("Ask any question to the chatbot (type 'x' to exit): ")
        if user_input != "x":
            state["messages"] = state.get("messages", []) + [
                {"role": "user", "content": user_input}
            ]
            state = graph.invoke(input=state)
            if state["messages"] and len(state["messages"]) > 0:
                latest_message = state["messages"][-1]
                print(f"\nAssistant: {latest_message.content}")

        else:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
