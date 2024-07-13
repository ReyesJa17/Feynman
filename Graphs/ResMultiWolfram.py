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



### Set the API keys
os.environ['GROQ_API_KEY'] 
os.environ['LANGCHAIN_API_KEY'] 
os.environ["LANGCHAIN_TRACING_V2"] 
os.environ["LANGCHAIN_PROJECT"] 
os.environ["WOLFRAM_ALPHA_APPID"]
os.environ["TAVILY_API_KEY"]



###Choose LLM model and provider

# Ollama model name
local_llm = "dolphin-mistral"

#llm = ChatOllama(model=local_llm, temperature=0)

llm = ChatGroq(model="llama3-70b-8192",temperature=0)


###Prompt templates




#Wolfram

prompt_wolfram = PromptTemplate(
    template=""""
    You are an assitant that rephrases a input formula to be more suitable for a Wolfram Alpha query. \n
    Here is the input: {input} \n
    Here are the rules to follow: \n
    - ALWAYS provide the response on a json with a single key 'input' and no preamble or explanation. \n
    - WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.\n
    - WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.\n
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
    -- Do not explain each step. Proceed directly to making a better API call based on the available assumptions.    
    """,
    input_variables=["input"],
)

#Prompt Formula

prompt_formula_wolfram_decomposition = PromptTemplate(
    template=
    """
    You are a helpful assistant. \n
    Your job is to analyze the question and return the main formula related to the question. \n
    Generate a query related to: {question} \n
    Return a json with the key 'formula' and the formula. \n
    
    """,
    input_variables=["question"],
)

#Prompt Multiquery 

prompt_multiquery_wolfram_decomposition = PromptTemplate(
    template=
    """
    You are a helpful assistant. \n
    Your job is to analyze the question and return queries concepts and formulas related to the question. \n
    only returnt the concepts separated by a comma. \n
    Generate multiple queries related to: {question} \n
    Return a json with the key 'queries' and a list with the queries separated by a comma. \n
    Each query must be one word. \n
    Output 3 queries. \n
    
    """,
    input_variables=["question"],
)


#Router
prompt_router = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore of Feynman lectures,query_wolframalpha or websearch. \n
    Use the vectorstore for questions about any theory question related to physics. \n
    You do not need to be stringent with the keywords in the question related to these topics. \n
    Always use wolframalpha for questions that require mathematical calculations, date and unit conversions, formula solving, etc. \n
    Just use the vectorstore for any question that does not require mathematical calculations. \n
    Otherwise, use web-search.
    Give a binary choice 'query_wolframalpha', 'web_search' or 'vectorstore' based on the question. \n
    Return the a JSON with a single key 'datasource' and no premable or explanation. \n
    Question to route: {question}""",
    input_variables=["question"],
)

### Retrieval Grader
prompt_retrival = PromptTemplate(
    template="""You are a grader assessing relevance of the related concepts to a user question. \n 
    Here are the related concepts: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the concepts are related to the user question and physics, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

### Hallucination Grader

prompt_hallucination = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)


### Answer Grader

prompt_grader = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)



#Generate

prompt_generate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for a physics student. Your job is to generate a response that contains the context, you dont need to answer the question just check the relation between the question and the context. \n"
            "Question: {question} "
           " Context: {context} "
            "Answer:"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


prompt_recursive_answer = PromptTemplate(
    template=
    """
    Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """,
    input_variables=["question", "q_a_pairs", "context"],
)





generate_queries_decomposition =  prompt_multiquery_wolfram_decomposition | llm | JsonOutputParser()



question_router = prompt_router | llm | JsonOutputParser()

retrieval_grader = prompt_retrival | llm | JsonOutputParser()

rag_chain_generate = prompt_generate | llm | StrOutputParser()

hallucination_grader = prompt_hallucination | llm | JsonOutputParser()

answer_grader = prompt_grader | llm | JsonOutputParser()


rag_chain_recursive = (
    {"context": itemgetter("question") ,
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | prompt_recursive_answer
    | llm
    | StrOutputParser())



wolfram_chain = prompt_wolfram | llm | JsonOutputParser()

chain_formula_decomposition = prompt_formula_wolfram_decomposition | llm | JsonOutputParser()

#Tools
web_search_tool = TavilySearchResults(k=1)

###GraphState

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    concepts: List[str]


###Functions


## Nodes 


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]



    print("---WOLFRAMALPHA---")
    response,concepts = query_wolframalpha_multiquery(question)

    


    return {"documents": response, "question": question, "concepts": concepts}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain_generate.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    concepts = state["concepts"]
    # Score each doc
    filtered_docs = []
    print(documents)
    for i in range(len(documents)):
        score = retrieval_grader.invoke(
            {"question": concepts[i], "document": documents[i]}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(documents)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    return {"documents": filtered_docs, "question": question}





### Edges ###


### Edges ###


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
        return "retrieve"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    



def query_wolframalpha_multiquery(input):
    """
        Uses the Wolfram Alpha API to query a question thorough a LLM

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
    """



    # Retrieve the API key from the environment variable
    appid = os.getenv("WOLFRAM_ALPHA_APPID")
    
    if not appid:
        raise ValueError("WOLFRAM_API_KEY environment variable is not set")
    info = []
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    res = generate_queries_decomposition.invoke({"question": input})
    list_concepts = res["queries"]
    for concept in list_concepts:
        print("----------------------")
        print(concept)
    



        rephrase_question= str(concept)
        # URL-encode the input query
        print(rephrase_question)
        encoded_query = urllib.parse.quote(rephrase_question)
        
        # Construct the full URL with the appid and input
        url = f"{base_url}?input={encoded_query}&appid={appid}"
        
        # Make the request to the API
        response = requests.get(url)
        print(response.text)

        info.append(response.text)
    return info, list_concepts
    
            


def query_wolframalpha(input):
    """
        Uses the Wolfram Alpha API to query a question thorough a LLM

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
    """



    # Retrieve the API key from the environment variable
    appid = os.getenv("WOLFRAM_ALPHA_APPID")
    
    if not appid:
        raise ValueError("WOLFRAM_API_KEY environment variable is not set")
    info = []
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"
    
    input = wolfram_chain.invoke({"input": input})
    print(input)
    encoded_query = urllib.parse.quote(input)
    
    # Construct the full URL with the appid and input
    url = f"{base_url}?input={encoded_query}&appid={appid}"
    
    # Make the request to the API
    response = requests.get(url)
    print(response.text)

    info.append(response.text)
    return info


def extract_suggestions(message):
    # Check if "Things to try instead:" is present in the message
    if "Things to try instead:" in message:
        try:
            # Locate the "Things to try instead:" part and extract the following text
            suggestions_part = message.split("Things to try instead:")[1]
            # Split the extracted text into individual suggestions based on newline separators
            suggestions = [suggestion.strip() for suggestion in suggestions_part.split("\n") if suggestion.strip()]
            return suggestions
        except IndexError:
            return []
    else:
        return "correct"





# ## Build Graph

workflow = StateGraph(GraphState)

# Define the nodes

workflow.add_node("retrieve", retrieve)  # retrieve vectorstore
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae




workflow.set_entry_point("retrieve")








#Vectorstore
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "retrieve": "retrieve",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "retrieve",
    }
)

def from_conn_stringx(cls, conn_string: str,) -> "SqliteSaver":
    return SqliteSaver(conn=sqlite3.connect(conn_string, check_same_thread=False))
#Memory
SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

memory = SqliteSaver.from_conn_stringx(":memory:")



# Compile
app = workflow.compile(checkpointer=memory)

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


#Tools ReAct





# Create the ReAct agent and chains


def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

    

def run_workflow_wolfram(inputs):
    for output in app.stream(inputs,config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
    return value["generation"]

  
def get_multiple_answer_wolfram(question: str):
    """
    Divides the question into multiple sub-questions and generates answers for each.

    Args:
        question (str): The input question
    """

    

    q_a_pairs = ""
    answer_context = run_workflow_wolfram({"question": question})
    
    answer = rag_chain_recursive.invoke({"question":question,"q_a_pairs":q_a_pairs,"context":answer_context})
    q_a_pair = format_qa_pair(question,answer_context)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    formula = chain_formula_decomposition.invoke({"question": question})
    print(formula)
 

    answer_formula = wolfram_chain.invoke({"input": formula["formula"]})
    answer = rag_chain_recursive.invoke({"question":question,"q_a_pairs":q_a_pairs,"context":answer_formula})
    q_a_pair = format_qa_pair(answer,answer_context)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    print(answer)
    
    return answer

def split_string_to_list(input_string):
    # Split the input string by commas and strip any leading/trailing whitespace from each element
    elements = [element.strip() for element in input_string.split(',')]
    return elements



