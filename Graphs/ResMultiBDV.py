
import uuid
from operator import itemgetter
from langchain.prompts import PromptTemplate
from pprint import pprint
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import sqlite3
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from typing import List
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from raptor_feynman import answer_raptor



### Set the API keys
os.environ['GROQ_API_KEY'] 
os.environ['LANGCHAIN_API_KEY'] 
os.environ["LANGCHAIN_TRACING_V2"] 
os.environ["LANGCHAIN_PROJECT"] 
os.environ["WOLFRAM_ALPHA_APPID"]




###Choose LLM model and provider

# Ollama model name
local_llm = "llama3.1"

llm_json = ChatOllama(model=local_llm, temperature=0)

llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
        )


###Prompt templates






#Prompt Multiquery 

prompt_steps_solve = PromptTemplate(
    template=
    """
    Your job is to generate a series of steps to solve a physics problem. \n
    The problem is: {question} \n
    Just provide the steps, do not solve the problem. \n
    """,
    input_variables=["question"],
)

prompt_multiquery_teoric_decomposition = PromptTemplate(
    template=
    """
    You are a helpful assistant that generates multiple sub-questions related to an input question and steps to solve a problem. \n
    The goal is to break down the input into a set of sub-questions that are related to the main question, this sub questions must be conceptually related to the main question. \n
    Dont answer the question, just generate the sub-questions. \n
    Generate multiple search queries related to: \m
    Question: \n
    {question} \n
    Steps to solve the problem: \n
    {steps}\n
    Provide a JSON list with at least 2 different sub-questions as strings. \n
    """,
    input_variables=["question", "steps"],
)

### Question Re-writer
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["question"],
)


### Retrieval Grader
prompt_retrival = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
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
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
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



generate_steps_to_solve = prompt_steps_solve | llm | StrOutputParser()

generate_queries_decomposition =  prompt_multiquery_teoric_decomposition | llm | JsonOutputParser()

question_rewriter = re_write_prompt | llm | StrOutputParser()

retrieval_grader = prompt_retrival | llm | JsonOutputParser()

rag_chain_generate = prompt_generate | llm | StrOutputParser()

hallucination_grader = prompt_hallucination | llm | JsonOutputParser()

answer_grader = prompt_grader | llm                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | JsonOutputParser()


rag_chain_recursive = (
    {"context": itemgetter("question") ,
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | prompt_recursive_answer
    | llm
    | StrOutputParser())




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

    # Retrieval
    documents = answer_raptor(question)
    return {"documents": documents, "question": question}


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

    # Score each doc
    filtered_docs = []
    print(documents)

    score = retrieval_grader.invoke(
        {"question": question, "document": documents}
    )
    grade = score["score"]
    if grade == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        filtered_docs.append(documents)
    else:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}




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
        return "transform_query"
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
    




# ## Build Graph

workflow = StateGraph(GraphState)

# Define the nodes

workflow.add_node("retrieve", retrieve)  # retrieve vectorstore
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query



workflow.set_entry_point("retrieve")








#Vectorstore
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
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




# Recursive RAG chain


def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

    

def run_workflow(inputs):
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

  
def get_multiple_answer(question: str):
    """
    Divides the question into multiple sub-questions and generates answers for each.

    Args:
        question (str): The input question
    """

    

    q_a_pairs = ""
    steps_to_solve = generate_steps_to_solve.invoke({"question": question})
    questions = generate_queries_decomposition.invoke({"question": answer_context, "steps": steps_to_solve})
    for q in questions:
        print(q)
        answer_context = run_workflow({"question": q})
        answer = rag_chain_recursive.invoke({"question":question,"q_a_pairs":q_a_pairs,"context":answer_context})
        q_a_pair = format_qa_pair(q,answer_context)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
        print(answer)
    
    return answer

