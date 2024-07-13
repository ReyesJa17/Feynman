from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
import uuid
from operator import itemgetter
from langchain.schema import Document
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.tools import  tool
from pprint import pprint
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import sqlite3
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from typing import List
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import requests
import urllib.parse
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from raptor_feynman import answer_raptor
from ResMultiBDV import get_multiple_answer 
from ResMultiWolfram import get_multiple_answer_wolfram


### Set the API keys
os.environ['GROQ_API_KEY'] 
os.environ['LANGCHAIN_API_KEY'] 
os.environ["LANGCHAIN_TRACING_V2"] 
os.environ["LANGCHAIN_PROJECT"] 
os.environ["WOLFRAM_ALPHA_APPID"]
os.environ["TAVILY_API_KEY"]



###Choose LLM model and provider

# Ollama model name
local_llm = "llama3:70b"

#llm = ChatOllama(model=local_llm, temperature=0)

llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
        )


###Prompt templates



#Prompt abstract main physics concepts

prompt_abstract_main_physics_concepts = PromptTemplate(
    template=
    """
    You are a physics assistant. \n
    Your job is to return all the the main concepts of physics and their close relatives. \n
    Return a JSON key "main_concepts" with a list of the main concepts of physics and their close relatives separated by a comma. \n
    Here is the problem: \n
    {problem}
    """,
    inputs=["problem"],
)

#Decide problem type

prompt_type_of_problem = PromptTemplate(
    template=
    """
    You are a physics assistant. \n
    Your job is to decide the closest category based on the physics problem. \n
    The categories types to choose are : Static, Kinematics, Dynamics, Energy, Momentum, Rotational Motion, Gravitation, Oscillations, Waves, Fluids, Thermodynamics, Electrostatics, Circuits, Magnetism, Optics, Modern Physics \n
    Return a JSON key "problem_type" with the type of problem. \n
    Here is the problem: \n
    {problem}
    """,
    inputs=["problem"],
)

#Obtain original formula

prompt_original_formula = PromptTemplate(
    template=
    """
    You are a physics assistant. \n
    Your job is to return the original formula of the given physics concept. \n
    Just return the formula without any substitutions. \n
    Return a JSON key "original_formula" with the original formula of the given physics concept. \n
    The type of problem is {problem_type}. \n
    Here is the problem: \n
    {problem}
    """,
    inputs=["problem", "problem_type"],
)

#Steps to solve the problem

prompt_steps_to_solve_problem = PromptTemplate(
    template=
    """
    You are a physics assistant. \n
    Based on the next physics problem, you need to return the steps to solve it. \n
    Return a JSON key "steps_to_solve" with the steps to solve the problemn separated by a comma. \n
    The type of problem is {problem_type}. \n
    Here is the problem: \n
    {problem}
    """,
    inputs=["problem", "problem_type"],
)   


#Unite final answer

prompt_unite_final_answer = PromptTemplate(
    template=
    """
    You are a physics assistant. \n
    Your job is to analyze all the information provided and return a final answer that not only solves the problem but also explains the solution. \n
    Leave any extras that might help the user understand the problem like examples or applications. \n
    Try just to sintetize the information provided so that the response is clear and concise. \n
    All details are important. \n
    Return a JSON key "final_answer" with the final answer. \n
    Here is the problem: \n
    {problem} \n
    The information at your disposal to answer is: \n
    Wolfram Alpha answer: {wolfram_alpha_answer} \n
    Vector data base answer: {vector_data_base_answer} \n
    Type of problem: {problem_type} \n

    
    """,
   inputs=["problem", "problem_type", "wolfram_alpha_answer", "vector_data_base_answer"],
)


#Prompt format answer
prompt_format_answer = PromptTemplate(
    template=
    """
    You are a helpful assistant. \n
    Your job is to format the final answer. \n
    The desire format must include first the original question summary, then the final answer, then some helpful applications or examples and at the end some curious facts. \n
    Just include information that is contained on the information provided. \n
    Here is the final answer to format: \n
    {final_answer}\n
    Here is the original question: \n
    {problem}\n
    Here are the helpful applications or examples: \n
    {applications_examples}\n
    Here are some curious facts: \n
    {curious_facts}\n
    """,
    inputs=["final_answer", "problem", "applications_examples", "curious_facts"],
)

#Prompt search web for curious facts
prompt_search_web_for_curious_facts = PromptTemplate(
    template=
    """
    You are a curious assistant. \n
    Your job is to transfor the problem into a search query and return the first result. \n
    The main goal is to find curious facts about the physics concept. \n
    Return a JSON key "curious_facts" with the curious facts. \n
    Here is the physics concept: \n
    {problem}
    """,
    inputs=["problem"],
)



#Chains


chain_abstract_main_physics_concepts = prompt_abstract_main_physics_concepts | llm | JsonOutputParser()
chain_type_of_problem = prompt_type_of_problem | llm | JsonOutputParser()
chain_original_formula = prompt_original_formula | llm | JsonOutputParser()
chain_steps_to_solve_problem = prompt_steps_to_solve_problem | llm | JsonOutputParser()
chain_unite_final_answer = prompt_unite_final_answer | llm | JsonOutputParser()
chain_format_answer = prompt_format_answer | llm | JsonOutputParser()
chain_search_web_for_curious_facts = prompt_search_web_for_curious_facts | llm | JsonOutputParser()


#Web search tool
web_search_tool = TavilySearchResults(k=1)




#Graph State

class GraphState(TypedDict):
    problem: str
    main_concepts: List[str]
    problem_type: str
    original_formula: str
    steps_to_solve: str
    wolfram_alpha_answer: str
    vector_data_base_answer: str
    final_answer: str



#Utility functions
 


def get_main_physics_concepts(state):
    """
    Retrieve the main concepts of physics and their close relatives.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the main concepts
    """

    problem = state["problem"]
    main_concepts = chain_abstract_main_physics_concepts.invoke({"problem": problem})
    print(main_concepts)
  
    return {"main_concepts": main_concepts["main_concepts"]}

def get_type_of_problem(state):
    """
    Decide the closest category based on the physics problem.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the type of problem
    """

    problem = state["problem"]
    problem_type = chain_type_of_problem.invoke({"problem": problem})
    print(problem_type)
  
    return {"problem_type": problem_type["problem_type"]}


def get_original_formula(state):
    """
    Retrieve the original formula of the given physics concept.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the original formula
    """

    problem = state["problem"]
    problem_type = state["problem_type"]
    original_formula = chain_original_formula.invoke({"problem": problem, "problem_type": problem_type})
  
    return {"original_formula": original_formula["original_formula"]}

def get_steps_to_solve_problem(state):
    """
    Retrieve the steps to solve the problem.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the steps to solve the problem
    """

    problem = state["problem"]
    problem_type = state["problem_type"]
    steps_to_solve = chain_steps_to_solve_problem.invoke({"problem": problem, "problem_type": problem_type})
    print(steps_to_solve)
  
    return {"steps_to_solve": steps_to_solve["steps_to_solve"]}

def unite_final_answer(state):
    """
    Analyze all the information provided and return a final answer that not only solves the problem but also explains the solution.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the final answer
    """

    problem = state["problem"]
    problem_type = state["problem_type"]
    wolfram_alpha_answer = get_multiple_answer_wolfram(problem)
    vector_data_base_answer = get_multiple_answer(problem)
    final_answer = chain_unite_final_answer.invoke({"problem": problem, "problem_type": problem_type, "wolfram_alpha_answer": wolfram_alpha_answer, "vector_data_base_answer": vector_data_base_answer})
    print(final_answer)
    return {"final_answer": final_answer["final_answer"]}

def get_curious_facts(state):
    """
    Retrieve curious facts about the physics concept.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the curious facts
    """

    problem = state["problem"]
    query = chain_search_web_for_curious_facts.invoke({"problem": problem})
    curious_facts = web_search_tool.invoke(({"query": query["curious_facts"]}))
    web_results = "\n".join([d["content"] for d in curious_facts])
    web_results = Document(page_content=web_results)
    print(web_results)

  
    return {"curious_facts": web_results}

#Graph
workflow_filter = StateGraph(GraphState)

workflow_filter.add_node("abstract_main_physics_concepts", get_main_physics_concepts)
workflow_filter.add_node("type_of_problem", get_type_of_problem)

workflow_filter.add_node("unite_final_answer", unite_final_answer)





workflow_filter.set_entry_point("abstract_main_physics_concepts")

workflow_filter.add_edge("abstract_main_physics_concepts", "type_of_problem")

workflow_filter.add_edge("type_of_problem", "unite_final_answer")

workflow_filter.add_edge("unite_final_answer", END)




def from_conn_stringx(cls, conn_string: str,) -> "SqliteSaver":
    return SqliteSaver(conn=sqlite3.connect(conn_string, check_same_thread=False))
#Memory
SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

memory = SqliteSaver.from_conn_stringx(":memory:")



# Compile
app = workflow_filter.compile(checkpointer=memory)

#config

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "patient_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


    

def run_workflow_filter(inputs):
    for output in app.stream(inputs,config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["final_answer"])
    return value["final_answer"]


