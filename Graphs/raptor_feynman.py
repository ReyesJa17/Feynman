

from langchain_community.embeddings import LlamaCppEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from typing import Dict, List, Optional, Tuple
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser




import getpass
import os
from langchain_groq import ChatGroq

RANDOM_SEED = 224  # Fixed seed for reproducibility


embd_model_path = r'C://Users//Reyes//Documents//NomicEmbed//nomic-embed-text-v1.5-GGUF//nomic-embed-text-v1.5.Q6_K.gguf'
# Embed and index
embedding = LlamaCppEmbeddings(model_path=embd_model_path, n_batch=1024)
os.environ['GROQ_API_KEY'] 

CHROMA_DIR = "feynman_storage"



def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
local_llm = "gemma2:27b-instruct-q6_K"
model = local_llm
#llm = ChatOllama(model=local_llm, temperature=0)

llm = ChatGroq( model="llama3-70b-8192",temperature=0)




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def chroma_db_exists():
    """
    Check if the Chroma database exists.

    Returns:
    - bool: True if the database exists, False otherwise.
    """
    return os.path.exists(CHROMA_DIR)


    # Check if Chroma database exists

def get_retriever():
    
    # Load existing Chroma vector store

    vectorstore_feynman = Chroma(

        embedding_function=embedding,
        persist_directory=CHROMA_DIR,
    )
    print("Loaded existing Chroma vector store")

    retriever = vectorstore_feynman.as_retriever()
    return retriever

print("Indexing complete")



def answer_raptor(question: str) -> str:
    retriever = get_retriever()



    prompt = hub.pull("rlm/rag-prompt")

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)




