import os
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_teddynote.prompts import load_prompt

# Load API KEY
load_dotenv()

st.title("Abby Chatbot")

with st.sidebar:
    clear_button = st.button("Clear Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask anything!")

if clear_button:
    st.write("Clearing chat...")
    st.session_state["messages"] = []

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        #st.write(f"{chat_message.role}: {chat_message.content}")
    

# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role = role, content = message))

def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a clerk in the city of abbotsford, BC"),
            ("user", "#Question:\n {question}")

        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    ouptputParser = StrOutputParser()

    chain = prompt | llm | ouptputParser

    return chain




if user_input:
    #print conversation on web
    print_messages()
    st.chat_message("user").write(user_input)
    chain = create_chain()
    #ai_answer = chain.invoke ({"question": user_input})
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
 
    

    #Record conversation
    add_message("user", user_input)
    add_message("assistant", ai_answer)

    
    # st.session_state["messages"].append(("user",user_input))
    # st.session_state["messages"].append(("assistant",user_input))
