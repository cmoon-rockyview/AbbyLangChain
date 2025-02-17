import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from sqlalchemy.engine import URL
from langchain.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Create a properly formatted URL
connection_url = URL.create(
    drivername="mssql+pyodbc",
    host="gdbdev",
    database="AbbyCC",
    query={"driver": "ODBC Driver 17 for SQL Server", "Trusted_Connection": "yes"},
)


# API KEY 정보로드
load_dotenv()


st.title("Search from Council Correspond!")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []


# 체인 생성
db = SQLDatabase.from_uri(str(connection_url))


def create_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = create_sql_query_chain(llm, db)

    prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )

    # model 은 gpt-3.5-turbo 를 지정
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # LLM 과 DB 를 매개변수로 입력하여 chain 을 생성합니다.
    execute_query = QuerySQLDataBaseTool(db=db)

    # SQL 쿼리 생성 체인
    write_query = create_sql_query_chain(llm, db)

    # 생성한 쿼리를 실행하기 위한 체인을 생성합니다.
    chain = write_query | execute_query

    answer = prompt | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    return chain
    #


chain = create_chain()

st.session_state["chain"] = create_chain()


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("Ask a question...")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        generated_sql_query = chain.invoke({"question": user_input})
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            chain = create_chain()

            result = chain.stream({"question": user_input})
            # container.markdown(result)

            ai_answer = ""
            for token in result:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
