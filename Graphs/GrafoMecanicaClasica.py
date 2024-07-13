import getpass
import os
import datetime as dt
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from datetime import datetime, date, timezone
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import logging
from typing import Optional, Union
from datetime import date, datetime
from langchain_core.output_parsers import StrOutputParser
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
import pandas as pd
from raptor_feynman import answer_raptor
from langchain_core.runnables import ensure_config
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
#logging.basicConfig(level=logging.DEBUG)
from typing import Callable
import requests
import urllib.parse
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
            model="llama3-70b-8192",
            temperature=0,
        )


#Prompts





#Prompt for generating final answer
prompt_unite_answer = PromptTemplate(
    template=""" You are a final answer generator. \n
    Your main goal is to create a concise, easy to understand and accurate answer to the user question. \n
    The answer must always be related to physics. \n
    If any of the context is not related to physics, do not include it in the final answer. \n
    Here is the question: {question} \n
    This is the wolframalpha solution to the question: {wolfram_solution} \n
    This is the vectorstore solution to the question: {vectorstore_solution} \n
    This is the websearch solution to the question: {websearch_solution} \n
    Provide the final answer to the question. \n
    Your main objective is to guide the user to the correct answer by explaining the main concepts related to the question. \n
    """,
    input_variables=["question", "wolfram_solution", "vectorstore_solution", "websearch_solution"],
)






#Problem divider

prompt_divide = PromptTemplate(
    template=
    """
    You are a physics assistant. Your job is to analyze a physics problem and provide a step-by-step solution. \n
    The answer must only contain the last formula with the values substituted to have the solution. \n
    Dont resolve the formula. \n
    If the problem does not have a solution, just give the closest formulas that are related. \n
    Return a json with the key 'solution' and the formula substitued as the solution. \n 
    The problem is: {problem}. \n
    """,
    input_variables=["problem"],
)



#Wolfram

prompt_wolfram = PromptTemplate(
    template=""""
    You are an assitant that rephrases a question to be more suitable for a Wolfram Alpha query. \n
    Here is the question: {question} \n
    Here are the rules to follow: \n
    - ALWAYS provide the response on a json with a single key 'input' and no preamble or explanation. \n
    - WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.\n
    - WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.\n
    - Convert inputs to simplified keyword queries whenever possible (e.g. convert "how many people live in France" to "France population").\n
    - Send queries in English only; translate non-English queries before sending, then respond in the original language.\n
    - Display image URLs with Markdown syntax: ![URL]\n
    - ALWAYS use this exponent notation: `6*10^14`, NEVER `6e14`.\n
    - ALWAYS use proper Markdown formatting for all math, scientific, and chemical formulas, symbols, etc.:  '$$\n[expression]\n$$' for standalone cases and '\( [expression] \)' when inline.\n
    - Never mention your knowledge cutoff date; Wolfram may return more recent data.\n
    - Use ONLY single-letter variable names, with or without integer subscript (e.g., n, n1, n_1).\n
    - Use named physical constants (e.g., 'speed of light') without numerical substitution.\n
    - Include a space between compound units (e.g., "Ω m" for "ohm*meter").\n
    - To solve for a variable in an equation with units, consider solving a corresponding equation without units; exclude counting units (e.g., books), include genuine units (e.g., kg).\n
    - If data for multiple properties is needed, make separate calls for each property.\n
    - If a WolframAlpha result is not relevant to the query:\n
    -- If Wolfram provides multiple 'Assumptions' for a query, choose the more relevant one(s) without explaining the initial result. If you are unsure, ask the user to choose.\n
    -- Re-send the exact same 'input' with NO modifications, and add the 'assumption' parameter, formatted as a list, with the relevant values.\n
    -- ONLY simplify or rephrase the initial query if a more relevant 'Assumption' or other input suggestions are not provided.\n
    -- Do not explain each step unless user input is needed. Proceed directly to making a better API call based on the available assumptions.    
    """,
    input_variables=["question"],
)


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
            "Your speciality topics are equilibrium, force momentum, and center of mass.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())



prompt_cinematic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for cinematic physics.\n"
            "Your speciality topics are time,mass,force,space,velocity,acceleration,position,displacement,uniformly accelerated rectilinear motion,unifomly rectilinear motion,Uniformly Accelerated Circular Motion and Uniform Circular Motion\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

prompt_movement_collisions = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for physics of movement and collisions.\n"
            "Your speciality topics are linear momentum, impulse,elastic collisions,conservation of momentum, inelastic collisions, and restitution coefficient.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_thermodynamics = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for thermodynamics.\n"
            "Your speciality topics are internal energy, heat, temperature, first law of thermodynamics, ideal gas equation,and heat capacity\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_fluid_mechanics = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for fluid mechanics.\n"
            "Your speciality topics are archimedes principle, pascal's principle, density, pressure, bernoulli's equation,continuity equation,stokes law,and reynolds number\n"
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
            "Your speciality topics are orbital motion, kepler's laws, newton's law of universal gravitation, gravitational field, and gravitational potential energy.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_oscillations = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for oscillations.\n"
            "Your speciality topics are simple pendulum, damped oscillations, forced oscillations, physical pendulum, mass-spring system,phase,period,amplitude, frequency, resonance and simple armomonic motion.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"

            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_work_and_energy = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for work and energy.\n"
            "Your speciality topics are work, kinetic energy, potential energy, elastic potential energy, power, cinematic energy, gravitational potential energy, and conservation of energy.\n"
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
            "Your speciality topics are first law of newton, second law of newton, third law of newton, friction force, tension, normal force, weight, friction coefficient, elastic constant and elastic force.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_rotation_and_angular_moment = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for advanced electromagnetism and optics.\n"
            "Your speciality topics are angular velocity, angular acceleration, angular momentum, torque, moment of inertia, and conservation of angular momentum.\n"
            "Use the provided tools to assist the user with their questions.\n"
            "The main objective is not to provide the answer directly but to guide the user to the correct answer.\n"
            "If no tool is needed, respond with a message to the user.\n"
            "If you need more information, ask the user for more details.\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())



#Chains



div_p_chain = prompt_divide | llm | JsonOutputParser()
wolfram_chain = prompt_wolfram | llm | JsonOutputParser()
unify_answer = prompt_unite_answer | llm | StrOutputParser()


#Tools
web_search_tool = TavilySearchResults(k=3)








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



def divide_problem(question):
    """
    Divide a problem into steps

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---DIVIDE PROBLEM---")

    res = div_p_chain.invoke({"problem": question})
    return res



def query_wolframalpha(question):
    """
        Uses the Wolfram Alpha API to query a question thorough a LLM

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
    """

    answer_wolfram = []

    # Retrieve the API key from the environment variable
    appid = os.getenv("WOLFRAM_ALPHA_APPID")
    
    if not appid:
        raise ValueError("WOLFRAM_API_KEY environment variable is not set")
    
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"


    rephrase_question=wolfram_chain.invoke({"question": question})
    rephrase_question= str(rephrase_question["input"])
    # URL-encode the input query
    print(rephrase_question)
    encoded_query = urllib.parse.quote(rephrase_question)
    
    # Construct the full URL with the appid and input
    url = f"{base_url}?input={encoded_query}&appid={appid}"
    
    # Make the request to the API
    response = requests.get(url)
    print(response.text)

    documents = response.text

    verify_answer = extract_suggestions(documents)
    if verify_answer == "correct":
        return documents
    else:
        for suggestion in verify_answer:
            suggestion = suggestion + " physics"
            url = f"{base_url}?input={suggestion}&appid={appid}"
            response = requests.get(url)
            answer_wolfram.append(response.text)
        return answer_wolfram
            




    
 
    
   


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
                "cinematic",
                "movement_collisions",
                "thermodynamics",
                "fluid_mechanics",
                "gravitation",
                "oscillations",
                "work_and_energy",
                "dynamics",
                "rotation_and_angular_moment",


                
            ]
        ],
        update_dialog_state,
    ]

# Tools

@tool
def web_search(question: str):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")


    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question_theoretical_problem": question}

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
    """Transfers work to a specialized assistant to handle any physics-related questions."""

    request: str = Field(
        description="The user's request for the Feynman assistant to handle a physics-related question."
        
    )



class Statics(BaseModel):
    """Transfers work to a specialized assistant to handle basic physics questions."""
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

class CinematicPhysics(BaseModel):
    """Transfers work to a specialized assistant to handle conservation laws and fundamental concepts."""

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

class MovementCollisions(BaseModel):
    """Transfers work to a specialized assistant to handle classical mechanics."""

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

class Thermodynamics(BaseModel):
    """Transfers work to a specialized assistant to handle special relativity."""

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
class FluidMechanics(BaseModel):
    """Transfers work to a specialized assistant to handle rotational motion and oscillations."""

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


class Gravitation(BaseModel):
    """Transfers work to a specialized assistant to handle waves and optics."""

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


class Oscillations(BaseModel):
    """Transfers work to a specialized assistant to handle advanced topics in waves and light."""

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

class WorkEnergy(BaseModel):
    """Transfers work to a specialized assistant to handle quantum mechanics and statistical mechanics."""

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

class Dynamics(BaseModel):
    """Transfers work to a specialized assistant to handle thermodynamics and heat engines."""

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

class RotationAngularMoment(BaseModel):
    """Transfers work to a specialized assistant to handle advanced electromagnetism and optics."""

    problem : str = Field( description="The question to solve")
    request: str = Field(
        description="Any necessary followup questions the advanced electromagnetism and optics assistant needs to ask the user."
    )

    class Config:
        schema_extra = {
            "example": {
                "problem": "How does electromagnetism work?",
                "request": "How does electromagnetism work?",
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
        CinematicPhysics,
        MovementCollisions,
        Thermodynamics,
        FluidMechanics,
        Gravitation,
        Oscillations,
        WorkEnergy,
        Dynamics,
        RotationAngularMoment,


    ]
)

static_runnable = prompt_static | llm.bind_tools(physics_tools + [CompleteOrEscalate])

cinematic_runnable = prompt_cinematic | llm.bind_tools(physics_tools + [CompleteOrEscalate])

movement_collisions_runnable = prompt_movement_collisions | llm.bind_tools(physics_tools + [CompleteOrEscalate])

thermodynamics_runnable = prompt_thermodynamics | llm.bind_tools(physics_tools + [CompleteOrEscalate])

fluid_mechanics_runnable = prompt_fluid_mechanics | llm.bind_tools(physics_tools + [CompleteOrEscalate])

gravitation_runnable = prompt_gravitation | llm.bind_tools(physics_tools + [CompleteOrEscalate])

oscillations_runnable = prompt_oscillations | llm.bind_tools(physics_tools + [CompleteOrEscalate])

work_and_energy_runnable = prompt_work_and_energy | llm.bind_tools(physics_tools + [CompleteOrEscalate])

dynamics_runnable = prompt_dynamics | llm.bind_tools(physics_tools + [CompleteOrEscalate])

rotation_and_angular_moment_runnable = prompt_rotation_and_angular_moment | llm.bind_tools(physics_tools + [CompleteOrEscalate])





 

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

# Cinematic physics assistant

builder.add_node("enter_cinematic", create_entry_node("cinematic", "cinematic"))
builder.add_node("cinematic", Assistant(cinematic_runnable))
builder.add_edge("enter_cinematic", "cinematic")
builder.add_node(
    "cinematic_tools",
    create_tool_node_with_fallback(physics_tools),
)

def route_cinematic(state: GraphState) -> Literal["cinematic_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "cinematic_tools"
builder.add_edge("cinematic_tools", "cinematic")
builder.add_conditional_edges("cinematic", route_cinematic)

# Movement collisions assistant

builder.add_node("enter_movement_collisions", create_entry_node("movement collisions", "movement_collisions"))
builder.add_node("movement_collisions", Assistant(movement_collisions_runnable))
builder.add_edge("enter_movement_collisions", "movement_collisions")
builder.add_node(
    "movement_collisions_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_movement_collisions(state: GraphState) -> Literal["movement_collisions_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "movement_collisions_tools"
builder.add_edge("movement_collisions_tools", "movement_collisions")
builder.add_conditional_edges("movement_collisions", route_movement_collisions)

# Thermodynamics assistant

builder.add_node("enter_thermodynamics", create_entry_node("thermodynamics", "thermodynamics"))
builder.add_node("thermodynamics", Assistant(thermodynamics_runnable))
builder.add_edge("enter_thermodynamics", "thermodynamics")
builder.add_node(
    "thermodynamics_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_thermodynamics(state: GraphState) -> Literal["thermodynamics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "thermodynamics_tools"
builder.add_edge("thermodynamics_tools", "thermodynamics")
builder.add_conditional_edges("thermodynamics", route_thermodynamics)

# Fluid mechanics assistant

builder.add_node("enter_fluid_mechanics", create_entry_node("fluid mechanics", "fluid_mechanics"))
builder.add_node("fluid_mechanics", Assistant(fluid_mechanics_runnable))
builder.add_edge("enter_fluid_mechanics", "fluid_mechanics")
builder.add_node(
    "fluid_mechanics_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_fluid_mechanics(state: GraphState) -> Literal["fluid_mechanics_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "fluid_mechanics_tools"
builder.add_edge("fluid_mechanics_tools", "fluid_mechanics")
builder.add_conditional_edges("fluid_mechanics", route_fluid_mechanics)

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

# Oscillations assistant

builder.add_node("enter_oscillations", create_entry_node("oscillations", "oscillations"))
builder.add_node("oscillations", Assistant(oscillations_runnable))
builder.add_edge("enter_oscillations", "oscillations")
builder.add_node(
    "oscillations_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_oscillations(state: GraphState) -> Literal["oscillations_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "oscillations_tools"
builder.add_edge("oscillations_tools", "oscillations")
builder.add_conditional_edges("oscillations", route_oscillations)

# Work and energy assistant

builder.add_node("enter_work_and_energy", create_entry_node("work and energy", "work_and_energy"))
builder.add_node("work_and_energy", Assistant(work_and_energy_runnable))
builder.add_edge("enter_work_and_energy", "work_and_energy")
builder.add_node(
    "work_and_energy_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_work_and_energy(state: GraphState) -> Literal["work_and_energy_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "work_and_energy_tools"
builder.add_edge("work_and_energy_tools", "work_and_energy")
builder.add_conditional_edges("work_and_energy", route_work_and_energy)

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

# Rotation and angular moment assistant

builder.add_node("enter_rotation_and_angular_moment", create_entry_node("rotation and angular moment", "rotation_and_angular_moment"))
builder.add_node("rotation_and_angular_moment", Assistant(rotation_and_angular_moment_runnable))
builder.add_edge("enter_rotation_and_angular_moment", "rotation_and_angular_moment")
builder.add_node(
    "rotation_and_angular_moment_tools",
    create_tool_node_with_fallback(physics_tools),
)
def route_rotation_and_angular_moment(state: GraphState) -> Literal["rotation_and_angular_moment_tools","leave_skill","__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "rotation_and_angular_moment_tools"
builder.add_edge("rotation_and_angular_moment_tools", "rotation_and_angular_moment")
builder.add_conditional_edges("rotation_and_angular_moment", route_rotation_and_angular_moment)


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
        elif tool_calls[0]["name"] == CinematicPhysics.__name__:
            return "enter_cinematic"
        elif tool_calls[0]["name"] == MovementCollisions.__name__:
            return "enter_movement_collisions"
        elif tool_calls[0]["name"] == Thermodynamics.__name__:
            return "enter_thermodynamics"
        elif tool_calls[0]["name"] == FluidMechanics.__name__:
            return "enter_fluid_mechanics"
        elif tool_calls[0]["name"] == Gravitation.__name__:
            return "enter_gravitation"
        elif tool_calls[0]["name"] == Oscillations.__name__:
            return "enter_oscillations"
        elif tool_calls[0]["name"] == WorkEnergy.__name__:
            return "enter_work_and_energy"
        elif tool_calls[0]["name"] == Dynamics.__name__:
            return "enter_dynamics"
        elif tool_calls[0]["name"] == RotationAngularMoment.__name__:
            return "enter_rotation_and_angular_moment"

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