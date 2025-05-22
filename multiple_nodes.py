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
    message_type: Literal["emotional", "logical"] = Field(
        default=...,
        description="Classify if the messages require an emotional(therapist) or logical response.",
    )


# Message State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


# Message Classifier Agent
def message_classify_agent(state: State) -> dict[str, Any]:
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

    return {"message_type": response.message_type}  # type: ignore


# Rout messages type
def router(state: State) -> dict[str, str]:
    message_type: str | None = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}


# Therapist Agent
def therapist_agent(state: State) -> dict[str, list[dict[str, Any]]]:
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
