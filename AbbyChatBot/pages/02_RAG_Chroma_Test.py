import streamlit as st
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from AbbyUtils import load_prompt, stream_response
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

st.title("RAG - Chroma Test 👓🕶🦺")
with st.sidebar:
    clear_btn = st.button("Clear Chat")


# st.title("AbbyGPT")
###https://docs.streamlit.io/develop/api-reference


def add_chat_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_chat_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def get_department(code: str) -> str:
    """
    Returns the corresponding Department/Service (DS) based on the input code.

    Parameters:
    code (str): The input code to look up.

    Returns:
    str: The corresponding Department/Service or 'Unknown Code' if not found.
    """
    ds_mapping = {
        "LLS": "Legal Legislative Service",
        "ABBYPD": "Abbotsford Police Department",
        "ISIR": "Innovation, Strategy & Intergovernment Relations",
        "PDS": "Planning and Development Service",
        "ENG": "Engineering",
        "Finance": "Finance and Procurement Services",
        "Airport": "Airport",
        "N/A": "Not Applicable",
        "PRC": "Parks Recreation and Culture",
        "OPS": "Operations",
        "Fire": "Fire Rescue Services",
    }

    return ds_mapping.get(code.replace(" ", ""), "Unknown Code")


def create_retrieve(query):
    # Define embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Define the path for ChromaDB persistence
    DB_PATH = r"D:\GISDev\AbbyLangChain\RAG\chroma_db"

    # Initialize Chroma vector store
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,  # Use the defined embeddings object
        collection_name="EHC24",
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


def create_chain_():

    prompt = ChatPromptTemplate(
        [
            ("system", "You are a kind chatbot"),
            ("user", "#Question:\n{email_content}"),
        ]
    )

    # GPT model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # output parser
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if clear_btn:
    st.session_state["messages"] = []

print_chat_message()
# user input
user_input = st.chat_input("Ask away!")

if user_input:
    st.chat_message("user").write(user_input)
    retrieve = create_retrieve(user_input)
    chain = create_chain(retrieve, model_name="gpt-4o-mini")

    response = chain.stream(user_input)
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # st.chat_message("assistant").write(ai_answer)

    add_chat_message("user", user_input)
    add_chat_message("assistant", ai_answer)
