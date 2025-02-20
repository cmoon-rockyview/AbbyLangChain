import os
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate , MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import load_prompt
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Load API KEY
load_dotenv()

st.title("Abby Chatbot with Memory")

with st.sidebar:
    clear_button = st.button("Clear Chat")
    selected_prompt = st.selectbox("Choose a prompt", ("Conversation", "Email SummaryM01", "Email SummaryM02" ))
    session_id = st.text_input("Input Session Id", "abc123")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # If the session ID is not in the store
        # Create a new ChatMessageHistory object and store it
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # Return the session history for the given session ID

user_input = st.chat_input("Ask anything!")

if clear_button:   
    st.session_state["messages"] = []
    
if "store" not in st.session_state:
    st.session_state["store"] = {}

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role = role, content = message))


def create_chain(prompt_type):

    prompt = ChatPromptTemplate.from_messages(
        [
            
            ("system", "You are a helpful assistant."),
            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),  
        ]
    )

    if prompt_type == "Email SummaryM01":
        st.session_state["messages"] = []        
        prompt = load_prompt("prompts/SummaryM01.yaml", encoding="utf-8")
    elif prompt_type == "Email SummaryM02":
        st.session_state["messages"] = []
        prompt = load_prompt("prompts/SummaryM02.yaml", encoding="utf-8")


    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    ouptputParser = StrOutputParser()

    chain = prompt | ChatOpenAI(model_name="gpt-4o-mini") | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # Set the function to retrieve chat history.
        input_messages_key="question",  # Set the key for input messages to "question".
        history_messages_key="chat_history",  # Set the key for chat history messages to "history".
        
    )    

    return chain_with_history

if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(selected_prompt)

if user_input:
    #print conversation on web
    chain = st.session_state["chain"]

    if chain is not None:

        print_messages()     
    
        st.chat_message("user").write(user_input)        

        response = chain.stream({"question": user_input}
                                , config={"configurable": {"session_id": session_id}})
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
            #Record conversation
            add_message("user", user_input)
            add_message("assistant", ai_answer)

    

