
import uuid

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


from langchain.prompts import ChatPromptTemplate

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

llm_json = ChatOllama(model=local_llm, temperature=0)

llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
        )


###Prompt templates





#Unite final answer

prompt_unite_final_answer = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help a student solve a physics problem. \n
    Your job is to analyze all the information provided and return a final answer that also explains the solution. \n
    Leave any extras that might help the user understand the problem like examples or applications. \n
    All details are important in order for the student to understand the problem. \n
    Return a JSON key "final_answer" with the final answer. \n
    Here is the problem: \n
    {problem} \n
    The information at your disposal to answer is: \n
    Vector data base information: {vector_data_base_answer} \n
    """,
   inputs=["problem" , "vector_data_base_answer"],
)


prompt_solve_physics_problem = ChatPromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help a student solve a physics problem. \n
    Your job is to analyze all the information provided and return a final answer. \n
    Leave any extras that might help the user understand the problem like examples or applications. \n
    Return a JSON key "final_answer" with the final answer. \n
    Here is the problem: \n
    {problem} \n
    Vector data base information: {vector_data_base_answer} \n
    """,
   inputs=["problem" ,  "vector_data_base_answer"],
)

prompt_translate_problem = ChatPromptTemplate(
    template=
    """
    You are a helpful translator\n
    Your job is to translate the answer from english to spanish\n
    Return a JSON key "translate" with the translation. \n
    Here is the answer to translate: \n
    {final_answer} \n
    """,
   inputs=["final_answer" ],
)




#Chains





chain_unite_final_answer = prompt_unite_final_answer | llm | JsonOutputParser()

chain_solve_physics_problem = prompt_solve_physics_problem | llm | JsonOutputParser()

chain_translate_problem = prompt_translate_problem | llm | JsonOutputParser()


#Graph State

class GraphState(TypedDict):
    problem: str


    steps_to_solve: str

    vector_data_base_answer: str
    final_answer: str
    translate: str



#Utility functions
 



def solve_physics_problem(state):
    """
    Analyze all the information provided and return a final answer.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the final answer
    """

    problem = state["problem"]
    final_answer = chain_solve_physics_problem.invoke({"problem": problem, "vector_data_base_answer": state["vector_data_base_answer"]})
    print(final_answer)
    return {"final_answer": final_answer["final_answer"]}


def unite_final_answer(state):
    """
    Analyze all the information provided and return a final answer that not only solves the problem but also explains the solution.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the final answer
    """

    problem = state["problem"]
    vector_data_base_answer = get_multiple_answer(problem)
    final_answer = chain_unite_final_answer.invoke({"problem": problem,  "vector_data_base_answer": vector_data_base_answer})
    print(final_answer)
    return {"final_answer": final_answer["final_answer"]}


def translate(state):
    """
    Translate the answer from english to spanish

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the translation
    """

    final_answer = state["final_answer"]
    translation = chain_translate_problem.invoke({"final_answer": final_answer})
    print(translation)
    return {"translate": translation["translate"]}


#Graph
workflow_filter = StateGraph(GraphState)




workflow_filter.add_node("vector_database_answer", unite_final_answer)


workflow_filter.add_node("solve_physics_problem",solve_physics_problem)

workflow_filter.add_node("translate_problem", translate)


workflow_filter.set_entry_point("vector_database_answer")

workflow_filter.add_edge("vector_database_answer",  "solve_physics_problem")

workflow_filter.add_edge("solve_physics_problem", "translate_problem")

workflow_filter.add_edge("translate", END)




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
    return value["translate"]


