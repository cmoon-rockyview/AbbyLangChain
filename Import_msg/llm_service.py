import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class EmailSummary(BaseModel):
    category: str = Field(description="The category of the email")
    reason_category: str = Field(description="Reason for the category")
    person: str = Field(description="The sender's name")
    subject: str = Field(description="The email subject")
    email_sender: str = Field(description="The sender's email address")
    email_recipient: str = Field(description="The recipient's email address")
    email_date: str = Field(description="The email's sent date")
    summary: str = Field(description="Summary of the email content")


def initialize_llm():
    """Initialize the LLM model and parser."""
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    parser = PydanticOutputParser(pydantic_object=EmailSummary)
    return llm, parser


def load_email_prompt(parser):
    """Loads the email summary prompt from a YAML file."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "prompt", "Email_Summary.yaml"
    )
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"The prompt file at {prompt_path} does not exist.")

    prompt_email = load_prompt(prompt_path, encoding="utf-8").partial(
        format=parser.get_format_instructions()
    )
    return prompt_email


def process_email_content(llm, parser, prompt_email, email_content):
    """Processes email content using the LLM model."""
    response = (prompt_email | llm).invoke({"email_content": email_content})
    return parser.parse(response.content)
