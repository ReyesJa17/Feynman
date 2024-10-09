
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
from ResMultiBDV import get_multiple_answer 
from ResMultiBDG import get_decision_contain, get_formulas_from_equality_relationships
from Calculator import solve_physics_equation_with_latex, solve_physics_equation_with_latex_list
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() 

### Set the API keys
os.environ['GROQ_API_KEY'] 
os.environ['LANGCHAIN_API_KEY'] 
os.environ["LANGCHAIN_TRACING_V2"] 
os.environ["LANGCHAIN_PROJECT"] 
os.environ['OPENAI_API_KEY'] 

api_key = os.environ["OPENAI_API_KEY"]



###Choose LLM model and provider

# Ollama model name
local_llm = "llama3:70b"

#llm_json = ChatOllama(model=local_llm, temperature=0)

llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            
        )


llm_json = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=api_key,
)

###Prompt templates


#Unite final answer

prompt_steps = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help a student solve a physics problem. \n
    Your job is to analyze all the information provided and return a set of steps to solve the problem. \n
    All details are important in order for the student to understand the problem. \n
    Dont solve the problem, just provide the steps. \n
    Return a list under the JSON key "steps" with the steps to solve the problem. \n
    Here is the problem: \n
    {problem} \n
    The information at your disposal to make the steps is: \n
    {vector_data_base_answer} \n
    """,
   inputs=["problem" , "vector_data_base_answer"],
)


prompt_solve_physics_problem = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help a student understand a physics problem. \n
    Your job is to analyze all the information provided and explain the problem solution, the steps, the formula used, important concept referenced on the problem and the solution. \n
    Add any extras that might help the user understand the problem like examples or applications. \n
    The answer must be between 50 and 100 words. \n
    Here is the problem: \n
    {problem} \n
    Vector data base information: {vector_data_base_answer} \n
    Steps to solve the problem: {steps_to_solve} \n
    Formula used: {formula} \n
    Here is the answer to the problem: \n
    {answer} \n
    """,
   inputs=["problem" ,  "vector_data_base_answer", "answer"],
)

prompt_translate_problem = PromptTemplate(
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

prompt_get_formula = PromptTemplate(
    template=
    """
    Your task is to analyze the query input and with that and the steps to solve the problem you must return the formula used on the specific step. \n
    Return a JSON key "formula" with the formula and a JSON key "desired_variable" with the desired variable to solve. \n
    Verify the formula is in a correct latex format. \n
    Use the previous steps and their solutions as reference to get the formula of the current step. \n
    Do not include any preamble or explanation, just return the JSON. \n
    Here is the query: \n
    {query} \n
    Here is the previous steps: \n
    {previous_steps} \n
    Here is the current step: \n
    {step} \n
    Here are the possible formulas: \n
    {formulas} \n

    """,
   inputs=["query", "step", "previous_steps", "formulas"],
)

prompt_sustitute_formula = PromptTemplate(
    template=
    """
    Your task is to analyze the query input and the desired formula to use. \n
    Based on this return a JSON with the key variables and under it a list with the variables and their values separated by a comma. \n
    Do not include any preamble or explanation, just return the JSON. \n
    Here is the query: \n
    {query} \n
    Here is the desired formula: \n
    {formula} \n

    """,
   inputs=["query", "formula"],
)

prompt_verify_latex_formula = PromptTemplate(
    template=
    """
    Your task is to analyze the formula input and verify if it is in the correct LaTeX format. Simplify the variable names by removing any unnecessary \text commands or complex structures. Ensure the formula is clean and in the correct LaTeX format.  \n
    Return a JSON key "formula" with the correct expression of the LaTeX formula. \n
    Do not include any preamble or explanation, just return the JSON. \n
    Here is the formula: \n
    {formula} \n

    """,
   inputs=["formula"],
)

#Chains

chain_sustitute_formula = prompt_sustitute_formula | llm_json | JsonOutputParser()

chain_get_formula = prompt_get_formula | llm_json | JsonOutputParser()

chain_steps = prompt_steps | llm | JsonOutputParser()

chain_solve_physics_problem = prompt_solve_physics_problem | llm_json | StrOutputParser()

chain_translate_problem = prompt_translate_problem | llm | JsonOutputParser()

chain_verify_latex_formula = prompt_verify_latex_formula | llm_json | JsonOutputParser()

#Graph State

class GraphState(TypedDict):
    problem: str


    steps_to_solve: List[str]
    formula: List[str]
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
    vector_data_base_answer = state["vector_data_base_answer"]
    steps_to_solve = state["steps_to_solve"]
    formulas = state["formula"]
    i = 0
    current_answer = ""
    for step in steps_to_solve:
        query = problem
        if i==0:
            step_answer = chain_get_formula.invoke({"query": query, "step": step, "previous_steps": "none", "formulas": formulas})
            i+=1
        else:
            step_answer = chain_get_formula.invoke({"query": query, "step": step, "previous_steps": current_answer, "formulas": formulas})
            i=i+1
        print(step_answer)
        formula = step_answer["formula"]
        final_formula = chain_verify_latex_formula.invoke({"formula": formula})
        print(final_formula)
        desired_variable = step_answer["desired_variable"]
        variables = chain_sustitute_formula.invoke({"query": query, "formula": final_formula["formula"]})
        print(variables["variables"])
        answer = solve_physics_equation_with_latex_list(formula, desired_variable, variables["variables"])
        current_answer = current_answer + "The step solution number: "+ str(i) + "\n" + "Answer to solve the step: " + str(answer) + "\n"
    print(current_answer)

    final_answer = chain_solve_physics_problem.invoke({"problem": problem, "vector_data_base_answer": vector_data_base_answer, "answer": current_answer, "steps_to_solve": steps_to_solve, "formula": formulas})
    print(final_answer)
    return {"final_answer": final_answer}


def get_steps(state):
    """
    Analyze all the information provided and return a final answer that not only solves the problem but also explains the solution.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the final answer
    """

    problem = state["problem"]
    concept,concepts = get_decision_contain(problem)
    print(concept)
    print(concepts)
    vector_data_base_answer = get_multiple_answer(problem, concepts)
    formula = get_formulas_from_equality_relationships(concept)
    steps = chain_steps.invoke({"problem": problem,  "vector_data_base_answer": vector_data_base_answer, "formula": formula})
    steps_list = steps["steps"]
    print(steps_list)
    print(formula)
    print(vector_data_base_answer)
    return {"steps_to_solve": steps_list, "vector_data_base_answer": vector_data_base_answer,"formula": formula}


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




workflow_filter.add_node("vector_database_answer", get_steps)


workflow_filter.add_node("solve_physics_problem",solve_physics_problem)



workflow_filter.set_entry_point("vector_database_answer")

workflow_filter.add_edge("vector_database_answer",  "solve_physics_problem")

workflow_filter.add_edge("solve_physics_problem", END)






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



