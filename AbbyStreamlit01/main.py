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

st.title("Email Summarizer 💬")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar
with st.sidebar:
    clear_btn = st.button("Clear Conversation")

# Print previous messages
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# Add new message
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

class EmailSummary(BaseModel):
    person: str = Field(description="Sender")
    company: str = Field(description="Sender's company")
    email: str = Field(description="Sender's email address")
    subject: str = Field(description="Email subject")
    summary: str = Field(description="Summary of the email body")
    date: str = Field(description="Date and time mentioned in the email body")

def create_email_parsing_chain():
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions.

    #QUESTION:
    Extract important information from {email_conversation}.

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """
    )

    prompt = prompt.partial(format=output_parser.get_format_instructions())

    chain = prompt | ChatOpenAI(model="gpt-4o-mini")

    return chain

def create_report_chain():
    prompt = load_prompt("prompts/email.yaml", encoding="utf-8")

    output_parser = StrOutputParser()

    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | output_parser

    return chain

if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("Ask anything!")

if user_input:
    st.chat_message("user").write(user_input)

    email_chain = create_email_parsing_chain()
    answer1 = email_chain.invoke({"email_conversation": user_input})
    parser = PydanticOutputParser(pydantic_object=EmailSummary)
    answer = parser.parse(answer1.content)

    params = {"engine": "google", "gl": "us", "hl": "en", "num": "3"}
    search = SerpAPIWrapper(params=params)
    search_query = f"{answer.person} {answer.email} in Abbotsford"
    search_result = search.run(search_query)
    search_result = eval(search_result)

    search_result_string = "\n".join(search_result)

    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result_string,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }

    response = report_chain.stream(report_chain_input)
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)
