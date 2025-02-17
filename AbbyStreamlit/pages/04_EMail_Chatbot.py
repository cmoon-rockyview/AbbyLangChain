import streamlit as st
from dotenv import load_dotenv

# LangChain / Abby imports
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from AbbyUtils import load_prompt
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()
# Streamlit app settings
st.title("Council Corresponds Chatbot 👓🕶🦺")
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
    selected_model = st.selectbox("Select a model", ("gpt-4o-mini", "gpt-4o"), index=0)

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


def create_retrieve(query):

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Define the path for FAISS database
    DB_PATH = r"D:\AbbyLangchain\RAG\faiss_db"

    # Load both FAISS indexes
    vectorstore_1 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC23"
    )
    vectorstore_2 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC24"
    )
    vectorstore_3 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC25"
    )
    # Merge the two FAISS indexes
    vectorstore_1.merge_from(vectorstore_2)  # This combines both indexes into one

    vectorstore_1.merge_from(vectorstore_3)  # This combines both indexes into one

    # Create a retriever from the merged FAISS vector store
    retriever = vectorstore_1.as_retriever(search_kwargs={"top_k": 3})

    return retriever


def create_chain(retriever, model_name="gpt-4o-mini"):
    prompt = load_prompt("prompts/Email_Chatbot.yaml", encoding="utf-8")

    # prompt = ChatPromptTemplate(
    #     [
    #         (
    #             "system",
    #             """You are an administrative assistant for the Mayor of Abbotsford.
    #             Your job is to answer questions from emails sent to Council Correspondence.
    #             The documents you access basically are all emails sent to Council Correspondence.
    #             Your name is Mushina.""",
    #         ),
    #         ("user", "#Question:\n{question}"),
    #     ]
    # )

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
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
