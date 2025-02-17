from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from sqlalchemy import create_engine
from langchain_google_genai import ChatGoogleGenerativeAI

# API key and project settings
load_dotenv()
logging.langsmith("EMAIL Report Chatbot")

# Streamlit app settings
st.title("Council Corresponds Report Agent 💬")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize list to store conversation


# Define constants
class MessageRole:
    """
    Define MessageRole
    """

    USER = "user"  # User message role
    ASSISTANT = "assistant"  # Assistant message role


class MessageType:
    """
    Define Message Type.
    """

    TEXT = "text"  # Text message
    FIGURE = "figure"  # Figure message
    CODE = "code"  # Code message
    DATAFRAME = "dataframe"  # DataFrame message


# Message related functions
def print_messages():
    """
    Display Message on Chatbot.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # Display text message
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # Display figure message
                    elif message_type == MessageType.CODE:
                        with st.status("Code Output", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # Display code message
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # Display DataFrame message
                else:
                    raise ValueError(f"Unknown content type: {content}")


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    Function to save a new message.

    Args:
        role (MessageRole): Message role (user or assistant)
        content (List[Union[MessageType, str]]): Message content
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend(
            [content]
        )  # Merge consecutive messages with the same role
    else:
        messages.append([role, [content]])  # Add new message with a new role


# Sidebar settings
with st.sidebar:
    clear_btn = st.button("Clear Conversation")  # Button to clear conversation
    # uploaded_file = st.file_uploader(
    #     "Please upload a CSV file.", type=["csv"], accept_multiple_files=True
    # )  # CSV file upload feature
    selected_model = st.selectbox(
        "Please select an OpenAI model.", ["gpt-4o-mini", "gpt-4o"], index=0
    )  # OpenAI model selection option
    apply_btn = st.button("Start Data Analysis")  # Button to start data analysis


# Callback functions
def tool_callback(tool) -> None:
    """
    Callback function to handle tool execution results.

    Args:
        tool (dict): Executed tool information
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:
                df_in_result = None
                with st.status("Analyzing data...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="Code Output", state="complete", expanded=False)

                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                return result
            else:
                st.error("DataFrame is not defined. Please upload a CSV file first.")
                return


def observation_callback(observation) -> None:
    """
    Callback function to handle observation results.

    Args:
        observation (dict): Observation results
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][
                1
            ].clear()  # Clear the last message in case of an error


def result_callback(result: str) -> None:
    """
    Callback function to handle final results.

    Args:
        result (str): Final result
    """
    pass  # Currently does nothing


# Agent creation function
def create_agent(dataframe, selected_model="gpt-4o-mini"):
    """
    Function to create a DataFrame agent.

    Args:
        dataframe (pd.DataFrame): DataFrame to analyze
        selected_model (str, optional): OpenAI model to use. Default is "gpt-4o"

    Returns:
        Agent: Created DataFrame agent
    """
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0),
        ##ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0),
        dataframe,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix="You are a professional data analyst and expert in Pandas. "
        "You must use Pandas DataFrame(`df`) to answer user's request. "
        "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
        "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference>\n"
        "- [IMPORTANT] Use `English` for your visualization title and labels."
        "- `muted` cmap, white background, and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n",
    )


# Question handling function
def ask(query):
    """
    Function to handle user's question and generate a response.

    Args:
        query (str): User's question
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant"):
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
            st.write(ai_answer)

        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


# Main logic
if clear_btn:
    st.session_state["messages"] = []  # Clear conversation

if apply_btn:

    # Database connection parameters
    server = "GDBDEV"
    database = "AbbyCC"
    table_name = "EHC"

    # Define connection string using SQLAlchemy
    params = urllib.parse.quote_plus(
        f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    # Read data into Pandas DataFrame
    query = f"SELECT Id, subject, Body , Category FROM {table_name}"
    df = pd.read_sql(query, engine)

    loaded_data = df  # Load CSV file
    st.session_state["df"] = loaded_data  # Save DataFrame
    st.session_state["python_tool"] = (
        PythonAstREPLTool()
    )  # Create Python execution tool
    st.session_state["python_tool"].locals[
        "df"
    ] = loaded_data  # Add DataFrame to Python execution environment
    st.session_state["agent"] = create_agent(
        loaded_data, selected_model
    )  # Create agent
    st.success("Settings are complete. Please start the conversation!")
elif apply_btn:
    st.warning("Please upload a file.")

print_messages()  # Display saved messages

user_input = st.chat_input("Ask anything you are curious about!")  # Get user input
if user_input:
    ask(user_input)  # Handle user question
