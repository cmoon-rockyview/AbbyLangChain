import os
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_teddynote.prompts import load_prompt

# Load API KEY
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask anything!")

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        #st.write(f"{chat_message.role}: {chat_message.content}")
    

# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role = role, content = message))
    

if user_input:
    #print conversation on web
    # st.chat_message("user").write(user_input)
    # st.chat_message("assistant").write("Thinking...")

    #Record conversation
    add_message("user", user_input)
    add_message("assistant", "Thinking...")

    print_messages()
    # st.session_state["messages"].append(("user",user_input))
    # st.session_state["messages"].append(("assistant",user_input))
