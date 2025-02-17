import os
from langchain_core.messages import AIMessageChunk
import yaml
import numpy as np
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts import loading
from langchain_core.prompts.base import BasePromptTemplate
import base64
import requests
from IPython.display import Image, display
import os


class MultiModal:
    def __init__(self, model, system_prompt=None, user_prompt=None):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant who helps users to write a report related to images"
        if self.user_prompt is None:
            self.user_prompt = "Explain the images as an alternative text."

    def encode_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image_content = response.content
            if url.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif url.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
        else:
            raise Exception("Failed to download image")

    def encode_image_from_file(self, file_path):
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif file_ext == ".png":
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"

    def encode_image(self, image_path):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return self.encode_image_from_url(image_path)
        else:
            return self.encode_image_from_file(image_path)

    def display_image(self, encoded_image):
        display(Image(url=encoded_image))

    def create_messages(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        encoded_image = self.encode_image(image_url)
        if display_image:
            self.display_image(encoded_image)

        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{encoded_image}"},
                    },
                ],
            },
        ]
        return messages

    def invoke(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.invoke(messages)
        return response.content

    def batch(
        self,
        image_urls: list[str],
        system_prompts: list[str] = [],
        user_prompts: list[str] = [],
        display_image=False,
    ):
        messages = []
        for image_url, system_prompt, user_prompt in zip(
            image_urls, system_prompts, user_prompts
        ):
            message = self.create_messages(
                image_url, system_prompt, user_prompt, display_image
            )
            messages.append(message)
        response = self.model.batch(messages)
        return [r.content for r in response]

    def stream(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.stream(messages)
        return response


def load_prompt(file_path, encoding="utf8") -> BasePromptTemplate:
    """
    Loads the prompt configuration based on the file path.

    This function reads the prompt configuration in YAML format from the given file path
    and loads the prompt according to the configuration.

    Parameters:
    file_path (str): The path to the prompt configuration file.

    Returns:
    object: The loaded prompt object.
    """
    with open(file_path, "r", encoding=encoding) as f:
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)


def stream_response(response, return_output=False):
    """
    Streams the response from the AI model, processing each chunk and printing it.

    This function iterates over each item in the `response` iterable. If an item is an instance of `AIMessageChunk`,
    it extracts and prints the content of the chunk. If an item is a string, it prints the string directly. Optionally,
    the function can return the concatenated string of all response chunks.

    Parameters:
    - response (iterable): An iterable of response chunks which can be `AIMessageChunk` objects or strings.
    - return_output (bool, optional): If True, the function returns the concatenated response string. Default is False.

    Returns:
    - str: If `return_output` is True, the concatenated response string. Otherwise, nothing is returned.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


def langsmith(project_name=None, set_enable=True):

    if set_enable:
        result = os.environ.get("LANGCHAIN_API_KEY")
        if result is None or result.strip() == "":
            print("LangChain API key is not set. ")
            return
        os.environ["LANGCHAIN_ENDPOINT"] = (
            "https://api.smith.langchain.com"  # LangSmith API Endpoint
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # true: Enable
        os.environ["LANGCHAIN_PROJECT"] = project_name  # Project Name
        print(f"LangSmith is enabled.\n[Project_Name]:{project_name}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # false: Disable
        print("LangSmith is disabled.")


def env_variable(key, value):
    os.environ[key] = value
