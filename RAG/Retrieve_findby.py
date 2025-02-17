import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def retrieve():
    # Define embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Define the path for ChromaDB persistence
    DB_PATH = r"D:\Work\Projects\RAG\chroma_db"

    # Initialize Chroma vector store
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,  # Use the defined embeddings object
        collection_name="EmailHistory",
    )
    retriever = db.as_retriever()
    return retriever


def create_chain(retriever, model_name="gpt-4o-mini"):

    prompt = load_prompt("prompts/EmailCat01.yaml", encoding="utf-8")

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {"context": retriever, "email_content": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


retrieve = retrieve()
chain = create_chain(retrieve, model_name="gpt-4o-mini")

query = """
Gas price is too high. Can you help me with that?
"""
response = chain.stream(query)

final_answer = ""
for token in response:
    final_answer += token
    print(token, end="", flush=True)
