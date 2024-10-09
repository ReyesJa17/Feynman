

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
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
RANDOM_SEED = 224  # Fixed seed for reproducibility
import time


os.environ['GROQ_API_KEY'] 
os.environ['OPENAI_API_KEY'] 
pinecone_api_key = os.environ['PINECONE_API_KEY']

pc = Pinecone(api_key=pinecone_api_key)

index_name = "thermo-raptor"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]


index = pc.Index(index_name)




CHROMA_DIR = "feynman_storage"



embeddings_pinecone = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=768)

vector_store = PineconeVectorStore(index=index, embedding=embeddings_pinecone)

llm = ChatGroq( model="llama3-70b-8192",temperature=0)




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)






def get_retriever_pinecone():
    retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
    )
    return retriever



def answer_raptor_pinecone(question: str) -> str:
    retriever = get_retriever_pinecone()



    prompt = hub.pull("rlm/rag-prompt")

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

