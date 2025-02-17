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
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

st.title("RAG - FAISS Test 👓🕶🦺")
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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Define the path for FAISS database
    DB_PATH = r"D:\GISDev\AbbyLangChain\RAG\faiss_db"

    # Load both FAISS indexes
    vectorstore_1 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC23"
    )
    vectorstore_2 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC24"
    )

    # Merge the two FAISS indexes
    vectorstore_1.merge_from(vectorstore_2)  # This combines both indexes into one

    # Create a retriever from the merged FAISS vector store
    retriever = vectorstore_1.as_retriever(search_kwargs={"top_k": 1})

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
