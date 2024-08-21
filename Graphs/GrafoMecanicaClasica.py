import getpass
import os

from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from datetime import datetime, date, timezone
from langgraph.prebuilt import ToolNode


from typing import Optional, Union
from datetime import date, datetime

#from langchain import  smith
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
import uuid
from langchain_core.messages.tool import ToolMessage
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Union, List, Any
from datetime import datetime, date, timezone
import os.path
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from raptor_feynman import answer_raptor
from langchain_core.runnables import ensure_config

from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
#logging.basicConfig(level=logging.DEBUG)
from typing import Callable

from pprint import pprint
from FilterRes import run_workflow_filter





def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

os.environ['GROQ_API_KEY'] 
os.environ['LANGCHAIN_API_KEY'] 
os.environ["LANGCHAIN_TRACING_V2"] 
os.environ["LANGCHAIN_PROJECT"] 
os.environ["WOLFRAM_ALPHA_APPID"]
os.environ["TAVILY_API_KEY"]



#Database

# Database details
db_file = "patients.sqlite"

# Connect to the new database (it will create the file if it does not exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Check if the table already exists
cursor.execute("""
SELECT name FROM sqlite_master WHERE type='table' AND name='medical_patients';
""")
table_exists = cursor.fetchone()

if not table_exists:
    # Create a new table for medical patients
    create_table_query = """
    CREATE TABLE medical_patients (
        Registro INTEGER PRIMARY KEY,
        Nombre TEXT NOT NULL,
        Edad INTEGER,
        Direccion TEXT,
        Telefono TEXT,
        Peso REAL,
        Correo TEXT
    );
    """
    cursor.execute(create_table_query)

    # Optional: Insert sample data into the new table
    sample_data = [
        (1, "John Doe", 30, "123 Main St", "555-1234", 70.5, "johndoe@example.com"),
        (2, "Jane Smith", 25, "456 Elm St", "555-5678", 65.2, "janesmith@example.com"),
        (3, "Alice Johnson", 35, "789 Oak St", "555-8765", 68.0, "alicejohnson@example.com"),
    ]

    insert_data_query = """
    INSERT INTO medical_patients (Registro, Nombre, Edad, Direccion, Telefono, Peso, Correo)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    cursor.executemany(insert_data_query, sample_data)

    print("The 'medical_patients' table has been created and updated with sample data.")
else:
    print("The 'medical_patients' table already exists.")

# Commit changes and close the connection
conn.commit()
conn.close()




llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
        )


#Prompts

prompt_manager = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your job is to delegate and let the user know the response to their question based on the tool message response recieved under the placeholder. \n"
            "If a user asks a physics-related question \n"
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool.\n"
            "Only the specialized assistants are given permission to do this for the user.\n"
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls.\n "
            "Do not question the answer just return it \n"
            "\n\nCurrent user information:\n<Student_info>\n{user_info}\n</Student_info>"
            "\nCurrent time: {time}.",
        ),  
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_static = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for statics physics.\n"
            "Your speciality topics are equilibrium, torque, center of mass, conditions of equilibrium and static rigis bodies\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())



prompt_kinematic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for kinematic physics.\n"
            "Your speciality topics are uniform motion, uniform accelerated motion, circular motion, projictile motion, kinematics of particles,, relative motion, frames of reference, and relative velocity.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

prompt_conservation_momentum = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for physics of conservation of momentum.\n"
            "Your speciality topics are linear momentum, collisions, conservation of linear momentum, system of particles, rocket propulsion, and impulse.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_rigid_body_dynamics = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for rigid body dynamics.\n"
            "Your speciality topics are rotational motion, moment of inertia, torque and angular acceleration, conservation of angular momentum, gyroscopic motion, rotational kinetic energy\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_lagrangian_mechanics = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for lagrangian mechanics.\n"
            "Your speciality topics are principle of least action, lagrange equations, applications of lagrangian mechanics,central forces, small oscillations\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_gravitation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for gravitation.\n"
            "Your speciality topics are universal law of gravitation, gravitational fields, motion of satellites, gravitational potential energy,keplers laws and orbital mechanics\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_oscillations_waves = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for oscillations and waves.\n"
            "Your speciality topics are simple harmonic motion, energy harmonic motion, damped oscillation, forced oscillations and resonance, pendulum, mechenical waves, principle of superposition, standing waves, waves propageation, sound waves\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_hamiltonian = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for hamiltonian mechanics.\n"
            "Your speciality topics are hamilton equations, canonical transformations, poisson brackets, hamilton-jacobi theory, action angle variables\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_dynamics = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for dynamics.\n"
            "Your speciality topics are first law of newton, second law of newton, third law of newton, applications of newtons law, friction, dynamics and circular motion, work and energy, conservation of energy, power, forces in non inertial frames\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_continuum_mechanics = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for continuum mechanics.\n"
            "Your speciality topics are stress and strain, elasticity, fluid mechanichs, viscosity, navier strokes equations\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())











#Utility functions

def extract_suggestions(message):
    # Check if "instead:" is present in the message
    if "instead:" in message:
        try:
            # Locate the "instead:" part and extract the following text
            suggestions_part = message.split("instead:")[1]
            # Split the extracted text into individual suggestions based on newline or space separators
            suggestions = [suggestion.strip() for suggestion in suggestions_part.split("\n") if suggestion.strip()]
            return suggestions
        except IndexError:
            return []
    else:
        # Return the original message if "instead:" is not found
        return "correct"




def retrieve(question):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")


    # Retrieval
    res = answer_raptor(question)
    return res




def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"








    
 
    
   


def update_dialog_state(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

def format_datetime(dt: Union[datetime, date]) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat() + 'Z'
    elif isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).isoformat() + 'Z'
    else:
        raise ValueError("Invalid datetime or date object")
    


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



def from_conn_stringx(cls, conn_string: str,) -> "SqliteSaver":
    return SqliteSaver(conn=sqlite3.connect(conn_string, check_same_thread=False))


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: GraphState) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the response to the main user problem is not complete until you use the tools to provide the final answer."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node





#Graph State

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    problem : str

    dialog_state: Annotated[
        list[
            Literal[
                "manager",
                "static",
                "kinematic",
                "conservation_momentum",
                "rigid_body_dynamics",
                "lagrangian_mechanics",
                "gravitation",
                "oscillations_waves",
                "hamiltonian",
                "dynamics",
                "continuum_mechanics",



                
            ]
        ],
        update_dialog_state,
    ]

# Tools

@tool 
def problem_solver(problem: str):
    """
    Solve a physics problem

    Args:
        problem (str): The physics problem to solve
    
    Returns:

        answer (str): The answer to the problem
    """

    


    print("---WOLFRAMALPHA---")
    response = run_workflow_filter({"problem": problem})
 


    print("The final answer is: ")
    
    print(response)


  
    

    return response




@tool
def fetch_medical_patient_information() -> list[dict]:
    """Fetch all medical patient information from the 'medical_patients' table.

    Returns:
        A list of dictionaries where each dictionary contains the details of a medical patient.
    """

    config = ensure_config()
    configuration = config.get("configurable", {})
    feynman_id = configuration.get("feynman_id", None)
    if not feynman_id:
        raise ValueError("No duck_id provided in the configuration.")



    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    query = """
    SELECT 
        Registro, Nombre, Edad, Direccion, Telefono, Peso, Correo
    FROM 
        medical_patients
    WHERE
        Registro = ?
    """
    cursor.execute(query, (feynman_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


#Classes


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: GraphState, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


class ClassicalMechanics(BaseModel):
    """Transfers work to a specialized assistant to handle any classical mechanics questions."""

    request: str = Field(
        description="The user's request for the Feynman assistant to handle a physics-related question."
        
    )



class Statics(BaseModel):
    """Transfers work to a specialized assistant to handle statics physics."""
    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="The user's request for the basic physics assistant to handle a physics-related question."
    )
    class Config:
        schema_extra = {
            "example": {
                "problem": "How to calculate distance?",
                "request": "How to calculate distance?",
            }
        }

class Dynamics(BaseModel):
    """Transfers work to a specialized assistant to handle dynamics physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="The user's request for the conservation laws assistant to handle a physics-related question."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "What is the conservation of energy?",
                "request": "What is the conservation of energy?",
            }
        }

class ConservationMomentum(BaseModel):
    """Transfers work to a specialized assistant to handle conservation of momentum questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="The user's request for the classical mechanics assistant to handle a physics-related question."
    )

    class Config:
        schema_extra = {
            "example": {                                                
                "problem": "What is the classical mechanics?",
                "request": "I want to ask a question about the classical mechanics.",                                                                       
            }
        }                                                                                           

class Gravitation(BaseModel):
    """Transfers work to a specialized assistant to handle gravitation physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the special relativity assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "What is the special relativity?",
                "request": "I want to ask a question about the special relativity.",
            }
        }
class OscillationsWaves(BaseModel):
    """Transfers work to a specialized assistant to handle oscillations and waves physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the rotational motion assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "what is the rotational motion?",
                "request": "what is the rotational motion?",
            }
        }


class RigidBodyDinamics(BaseModel):
    """Transfers work to a specialized assistant to handle rigid body dynamics physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the waves and optics assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "How does the waves work?",
                "request": "How does the waves work?",
            }
        }


class LagrangianMechanics(BaseModel):
    """Transfers work to a specialized assistant to handle lagrangian mechanics physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the advanced topics in waves and light assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "How does the light work?",
                "request": "How does the light work?",
            }
        }

class HamiltonianMechanics(BaseModel):
    """Transfers work to a specialized assistant to handle hamiltonian mechanics physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the quantum mechanics and statistical mechanics assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "What is the brownian motion?",
                "request": "What is the brownian motion?",
            }
        }

class ContinuumMechanics(BaseModel):
    """Transfers work to a specialized assistant to handle continuum mechanics physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the thermo heat engines assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "What is the heat engines?",
                "request": "What is the heat engines?",
            }
        }

class Kinematics(BaseModel):
    """Transfers work to a specialized assistant to handle kinematics physics questions."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the kinematics assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "What is the kinematics?",
                "request": "What is the kinematics?",
            }
        }

# Set up the tools

physics_tools = [
    problem_solver,

]



manager_tools = [

]

# Runnable

manager_runnable = prompt_manager | llm.bind_tools(
    manager_tools
    +[
        ClassicalMechanics,
        Statics,
        Dynamics,
        ConservationMomentum,
        Gravitation,
        OscillationsWaves,
        RigidBodyDinamics,
        LagrangianMechanics,
        HamiltonianMechanics,
        ContinuumMechanics,


    ]
)

static_runnable = prompt_static | llm.bind_tools(physics_tools + [CompleteOrEscalate])

kinematics_runnable = prompt_kinematic | llm.bind_tools(physics_tools + [CompleteOrEscalate])

conservation_momentum_runnable = prompt_conservation_momentum | llm.bind_tools(physics_tools + [CompleteOrEscalate])

continuum_mechanics_runnable = prompt_continuum_mechanics | llm.bind_tools(physics_tools + [CompleteOrEscalate])

hamiltonian_runnable = prompt_hamiltonian | llm.bind_tools(physics_tools + [CompleteOrEscalate])

gravitation_runnable = prompt_gravitation | llm.bind_tools(physics_tools + [CompleteOrEscalate])

oscillation_waves_runnable = prompt_oscillations_waves | llm.bind_tools(physics_tools + [CompleteOrEscalate])

lagrangian_mechanics_runnable = prompt_lagrangian_mechanics | llm.bind_tools(physics_tools + [CompleteOrEscalate])

dynamics_runnable = prompt_dynamics | llm.bind_tools(physics_tools + [CompleteOrEscalate])

rigid_body_dynamics_runnable = prompt_rigid_body_dynamics | llm.bind_tools(physics_tools + [CompleteOrEscalate])





 

# Graph

builder = StateGraph(GraphState)


def user_info(state:GraphState):
    return{"user_info": fetch_medical_patient_information.invoke({})}


# Define entry
builder.add_node("fetch_user_info", user_info)
builder.set_entry_point("fetch_user_info")

# Statics assistant

builder.add_node("enter_static", create_entry_node("static", "static"))
builder.add_node("static", Assistant(static_runnable)) 
builder.add_edge("enter_static", "static")
builder.add_node(
    "static_tools",
    create_tool_node_with_fallback(physics_tools),
)

def route_static(state: GraphState) -> Literal["static_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "static_tools"

builder.add_edge("static_tools", "static")
builder.add_conditional_edges("static", route_static)

# Rigid Body Dynamics assistant

builder.add_node("enter_rigid_body_dynamics", create_entry_node("rigid_body_dynamics", "rigid_body_dynamics"))
builder.add_node("rigid_body_dynamics", Assistant(rigid_body_dynamics_runnable))
builder.add_edge("enter_rigid_body_dynamics", "rigid_body_dynamics")
builder.add_node(
    "rigid_body_dynamics_tools",
    create_tool_node_with_fallback(physics_tools),
)

def route_rigid_body_dynamics(state: GraphState) -> Literal["rigid_body_dynamics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "rigid_body_dynamics_tools"
builder.add_edge("rigid_body_dynamics_tools", "rigid_body_dynamics")
builder.add_conditional_edges("rigid_body_dynamics", route_rigid_body_dynamics)

# Conservation of Momentum assistant

builder.add_node("enter_conservation_momentum", create_entry_node("conservation_momentum", "conservation_momentum"))
builder.add_node("conservation_momentum", Assistant(conservation_momentum_runnable))
builder.add_edge("enter_conservation_momentum", "conservation_momentum")
builder.add_node(
    "conservation_momentum_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_conservation_momentum(state: GraphState) -> Literal["conservation_momentum_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "conservation_momentum_tools"
builder.add_edge("conservation_momentum_tools", "conservation_momentum")
builder.add_conditional_edges("conservation_momentum", route_conservation_momentum)

# Oscillations and Waves assistant

builder.add_node("enter_oscillation_waves", create_entry_node("oscillation_waves", "oscillation_waves"))
builder.add_node("oscillation_waves", Assistant(oscillation_waves_runnable))
builder.add_edge("enter_oscillation_waves", "oscillation_waves")
builder.add_node(
    "oscillation_waves_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_oscillation_waves(state: GraphState) -> Literal["oscillation_waves_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "oscillation_waves_tools"
builder.add_edge("oscillation_waves_tools", "oscillation_waves")
builder.add_conditional_edges("oscillation_waves", route_oscillation_waves)

# Lagrangian Mechanics assistant

builder.add_node("enter_lagrangian_mechanics", create_entry_node("lagrangian_mechanics", "lagrangian_mechanics"))
builder.add_node("lagrangian_mechanics", Assistant(lagrangian_mechanics_runnable))
builder.add_edge("enter_lagrangian_mechanics", "lagrangian_mechanics")
builder.add_node(
    "lagrangian_mechanics_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_lagrangian(state: GraphState) -> Literal["lagrangian_mechanics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "fluid_mechanics_tools"
builder.add_edge("lagrangian_mechanics_tools", "lagrangian_mechanics")
builder.add_conditional_edges("lagrangian_mechanics", route_lagrangian)

# Gravitation assistant

builder.add_node("enter_gravitation", create_entry_node("gravitation", "gravitation"))
builder.add_node("gravitation", Assistant(gravitation_runnable))
builder.add_edge("enter_gravitation", "gravitation")
builder.add_node(   
    "gravitation_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_gravitation(state: GraphState) -> Literal["gravitation_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "gravitation_tools"
builder.add_edge("gravitation_tools", "gravitation")
builder.add_conditional_edges("gravitation", route_gravitation)

# Hamiltonian Mechanics assistant

builder.add_node("enter_hamiltonian", create_entry_node("hamiltonian", "hamiltonian"))
builder.add_node("hamiltonian", Assistant(hamiltonian_runnable))
builder.add_edge("enter_hamiltonian", "hamiltonian")
builder.add_node(
    "hamiltonian_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_hamiltonian(state: GraphState) -> Literal["hamiltonian_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "hamiltonian_tools"
builder.add_edge("hamiltonian_tools", "hamiltonian")
builder.add_conditional_edges("hamiltonian", route_hamiltonian)

# Continuum Mechanics assistant

builder.add_node("enter_continuum_mechanics", create_entry_node("continuum_mechanics", "continuum_mechanics"))
builder.add_node("continuum_mechanics", Assistant(continuum_mechanics_runnable))
builder.add_edge("enter_continuum_mechanics", "continuum_mechanics")
builder.add_node(
    "continuum_mechanics_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_continuum_mechanics(state: GraphState) -> Literal["continuum_mechanics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "continuum_mechanics_tools"
builder.add_edge("continuum_mechanics_tools", "continuum_mechanics")
builder.add_conditional_edges("continuum_mechanics", route_continuum_mechanics)

# Dynamics assistant

builder.add_node("enter_dynamics", create_entry_node("dynamics", "dynamics"))
builder.add_node("dynamics", Assistant(dynamics_runnable))
builder.add_edge("enter_dynamics", "dynamics")
builder.add_node(
    "dynamics_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_dynamics(state: GraphState) -> Literal["dynamics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "dynamics_tools"
builder.add_edge("dynamics_tools", "dynamics")
builder.add_conditional_edges("dynamics", route_dynamics)

# Kinematics assistant

builder.add_node("enter_kinematics", create_entry_node("kinematics", "kinematics"))
builder.add_node("kinematics", Assistant(kinematics_runnable))
builder.add_edge("enter_kinematics", "kinematics")
builder.add_node(
    "kinematics_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_kinematics(state: GraphState) -> Literal["kinematics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "kinematics_tools"
builder.add_edge("kinematics_tools", "kinematics")
builder.add_conditional_edges("kinematics", route_kinematics)


def pop_dialog_state(state: GraphState) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.If the answer is on the conversation then return it as it is.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "manager")




# Manager

builder.add_node("manager", Assistant(manager_runnable))
builder.add_node("manager_tools", create_tool_node_with_fallback(manager_tools))

def route_manager(state: GraphState) -> Literal["manager_tools","enter_static","enter_cinematic","enter_movement_collisions","enter_thermodynamics","enter_fluid_mechanics","enter_gravitation","enter_oscillations","enter_work_and_energy","enter_dynamics","enter_rotation_and_angular_moment", "__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == Statics.__name__:
            return "enter_static"
        elif tool_calls[0]["name"] == Kinematics.__name__:
            return "enter_kinematics"
        elif tool_calls[0]["name"] == ConservationMomentum.__name__:
            return "enter_conservation_momentum"
        elif tool_calls[0]["name"] == ContinuumMechanics.__name__:
            return "enter_continuum_mechanics"
        elif tool_calls[0]["name"] == HamiltonianMechanics.__name__:
            return "enter_hamiltonian"
        elif tool_calls[0]["name"] == Gravitation.__name__:
            return "enter_gravitation"
        elif tool_calls[0]["name"] == OscillationsWaves.__name__:
            return "enter_oscillation_waves"
        elif tool_calls[0]["name"] == RigidBodyDinamics.__name__:
            return "enter_rigid_body_dynamics"
        elif tool_calls[0]["name"] == Dynamics.__name__:
            return "enter_dynamics"
        elif tool_calls[0]["name"] == LagrangianMechanics.__name__:
            return "enter_lagrangian_mechanics"

        return "manager_tools"
    raise ValueError("Invalid Route")


builder.add_conditional_edges(
    "manager",
    route_manager,
    {
        "enter_static": "enter_static",
        "enter_cinematic": "enter_cinematic",
        "enter_movement_collisions": "enter_movement_collisions",
        "enter_thermodynamics": "enter_thermodynamics",
        "enter_fluid_mechanics": "enter_fluid_mechanics",
        "enter_gravitation": "enter_gravitation",
        "enter_oscillations": "enter_oscillations",
        "enter_work_and_energy": "enter_work_and_energy",
        "enter_dynamics": "enter_dynamics",
        "enter_rotation_and_angular_moment": "enter_rotation_and_angular_moment",
        "manager_tools": "manager_tools",
        END: END,
    },
)
builder.add_edge("manager_tools", "manager")


def route_workflow(state: GraphState) -> Literal["manager","static","cinematic","movement_collisions","thermodynamics","fluid_mechanics","gravitation","oscillations","work_and_energy","dynamics","rotation_and_angular_moment" ]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "manager"
    return dialog_state[-1]

builder.add_conditional_edges("fetch_user_info", route_workflow)
   

SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)



memory = SqliteSaver.from_conn_stringx(":memory:")

part_1_graph = builder.compile(checkpointer=memory)

#Run Test

tutorial_questions = [
    "hi",
    "Is the doctor free for an appointment on may 30 at 9 am?",
    "Great! Can you book that appointment for me?",
    "Thanks",
]



thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "feynman_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}



def run_multiple_questions():

    _printed = set()
    for question in tutorial_questions:
        events = part_1_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)


def get_response (question):
    _printed = set()
    events = part_1_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    return event.get("messages")[-4].content

get_response("What is the conservation of energy?")