from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from langchain_core.output_parsers import JsonOutputParser
import sqlite3
import uuid
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from typing import List
from typing import Annotated
from langchain_core.tools import tool
from datetime import date, datetime
from typing import Optional, Union
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing_extensions import TypedDict
from FilterRes import run_workflow_filter
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]




### Set the API keys
os.environ['GROQ_API_KEY'] 





llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
        )


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful teacher for physics."
            "Use the provided tools to best answer all the user physic related questions. "
            "Always use the tools to answer physics related questions. "
            "If the user greets you, greet back. "
            "Answer in spanish."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }



def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )




@tool
def solve_problem(question:str):
    """
    Uses the vectorstore to search for the question answer.

    Args:
        question (str): The question to search for.

    Returns:
        str: The tool answer unchanged.
    """
    answer = run_workflow_filter({"problem": question})
    return answer


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    

part_1_tools = [
    solve_problem,
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)



builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")


memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)



thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "userr_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}



def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def get_response (question):
    _printed = set()
    i=0 
    events = part_1_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
        i = i+1
    if i>2:
        return event.get("messages")[-2].content
    else:
        return event.get("messages")[-1].content

#res = get_response("Porque el cielo es azul?")
