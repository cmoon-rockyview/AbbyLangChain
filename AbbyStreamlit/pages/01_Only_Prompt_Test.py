import streamlit as st
import numpy as np
from dotenv import load_dotenv

# LangChain / Abby imports
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from AbbyUtils import load_prompt

# Load environment variables from .env file
load_dotenv()
st.title("Only Prompt Test 👓🕶🦺")
# ------------------------------------------------------------------------------
# 1. Initialize session state to store chat messages
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ------------------------------------------------------------------------------
# 2. Sidebar UI: Clear chat button, prompt selection, and model selection
# ------------------------------------------------------------------------------
with st.sidebar:
    clear_btn = st.button(
        "Clear Chat", help="Clear all messages from the chat history."
    )
    selected_prompt = st.selectbox(
        "Select a prompt",
        ("Normal Conversation", "Prompt1", "Prompt2", "Prompt3"),
        index=0,
    )
    selected_model = st.selectbox(
        "Select a model", ("gpt-4o-mini", "gemini-1.5-pro", "gpt-4o"), index=0
    )

# Clear the session state if the user clicks the "Clear Chat" button
if clear_btn:
    st.session_state["messages"] = []


# ------------------------------------------------------------------------------
# 3. Helper Functions
# ------------------------------------------------------------------------------
def add_chat_message(role: str, message: str) -> None:
    """
    Append a new message to the session state.
    :param role: 'user' or 'assistant' or 'system'.
    :param message: The content of the chat message.
    """
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_chat_message() -> None:
    """
    Render each chat message in the Streamlit UI.
    """
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def create_chain(prompt_type: str):
    """
    Create and return a chain object based on the selected prompt type and model.
    1. Builds the prompt template.
    2. Chooses the LLM (Language Model) based on user selection.
    3. Constructs a pipeline of (PromptTemplate -> LLM -> OutputParser).
    """
    # Default prompt template (used for "Normal Conversation")
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are an administrative assistant for the Mayor of Abbotsford.",
            ),
            ("user", "#Question:\n{email_content}"),
        ]
    )

    # Load specific prompt if user selected a specialized one
    if prompt_type == "Prompt1":
        prompt = load_prompt("prompts/Cat01.yaml")
    elif prompt_type == "Prompt2":
        prompt = load_prompt("prompts/Cat02.yaml")
    elif prompt_type == "Prompt3":
        prompt = load_prompt("prompts/Cat03.yaml")  # Adjust path if needed

    # Choose the language model (LLM)
    if selected_model == "gpt-4o-mini":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif selected_model == "gemini-1.5-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    else:  # selected_model == "gpt-4o"
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Output parser
    output_parser = StrOutputParser()

    # Build the chain: prompt -> llm -> parser
    chain = prompt | llm | output_parser
    return chain


# ------------------------------------------------------------------------------
# 4. Print existing chat messages (if any) at the start
# ------------------------------------------------------------------------------
print_chat_message()

# ------------------------------------------------------------------------------
# 5. Main Chat Flow - Handle user input, generate AI response, and display
# ------------------------------------------------------------------------------
user_input = st.chat_input("Ask away!")

if user_input:
    # Display the user's message
    st.chat_message("user").write(user_input)

    # Create a chain using the selected prompt type
    chain = create_chain(selected_prompt)

    # Stream the response token-by-token
    response = chain.stream({"email_content": user_input})

    # Render the AI response in a chat message with real-time streaming
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # Save both user and assistant messages to session state
    add_chat_message("user", user_input)
    add_chat_message("assistant", ai_answer)
