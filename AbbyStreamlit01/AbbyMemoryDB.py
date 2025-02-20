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


# Load API KEY
load_dotenv()

st.title("Abby Chatbot with Memory database")

with st.sidebar:
    clear_button = st.button("Clear Chat")
    selected_prompt = st.selectbox("Choose a prompt", ("Conversation", "Email SummaryM01", "Email SummaryM02" ))


# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask anything!")

if clear_button:   
    st.session_state["messages"] = []

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role = role, content = message))

def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )

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
    elif prompt_type == "Email Summary02":
        st.session_state["messages"] = []
        prompt = load_prompt("prompts/SummaryM02.yaml", encoding="utf-8")


    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    ouptputParser = StrOutputParser()

    chain = prompt | ChatOpenAI(model_name="gpt-4o-mini") | StrOutputParser()

    config_fields = [
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for a user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for a conversation.",
            default="",
            is_shared=True,
        ),
    ]

    config = {"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_chat_history,  # Set the function to retrieve chat history.
        input_messages_key="question",  # Set the key for input messages to "question".
        history_messages_key="chat_history",  # Set the key for chat history messages to "history".
        history_factory_config=config_fields,  # Set the parameters to refer to when retrieving chat history.
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
                                , config={"user_id": "user1", "conversation_id": "conversation1"})
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
            #Record conversation
            add_message("user", user_input)
            add_message("assistant", ai_answer)

    

