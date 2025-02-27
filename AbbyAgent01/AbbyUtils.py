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
import feedparser
import urllib.parse
from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid

# 최종 결과를 출력하는 콜백 함수입니다.
def result_callback(result: str) -> None:
    print("[최종 답변]")
    print(result)  # 최종 답변을 출력합니다.

# 관찰 결과를 출력하는 콜백 함수입니다.
def observation_callback(observation) -> None:
    print("[관찰 내용]")
    print(f"Observation: {observation.get('observation')}")  # 관찰 내용을 출력합니다.

# 도구 호출 시 실행되는 콜백 함수입니다.
def tool_callback(tool) -> None:
    print("[도구 호출]")
    print(f"Tool: {tool.get('tool')}")  # 사용된 도구의 이름을 출력합니다.
    if tool_input := tool.get("tool_input"):  # 도구에 입력된 값이 있다면
        for k, v in tool_input.items():
            print(f"{k}: {v}")  # 입력값의 키와 값을 출력합니다.
    print(f"Log: {tool.get('log')}")  # 도구 실행 로그를 출력합니다.


@dataclass
class AgentCallbacks:
    """
    에이전트 콜백 함수들을 포함하는 데이터 클래스입니다.

    Attributes:
        tool_callback (Callable[[Dict[str, Any]], None]): 도구 사용 시 호출되는 콜백 함수
        observation_callback (Callable[[Dict[str, Any]], None]): 관찰 결과 처리 시 호출되는 콜백 함수
        result_callback (Callable[[str], None]): 최종 결과 처리 시 호출되는 콜백 함수
    """

    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback

class AgentStreamParser:
    """
    A class for parsing and processing the stream output of an agent.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        """
        Initializes the AgentStreamParser object.

        Args:
            callbacks (AgentCallbacks, optional): Callback functions to use during parsing. Defaults to AgentCallbacks().
        """
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        Processes the steps of the agent.

        Args:
            step (Dict[str, Any]): The agent step information to process.
        """
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        """
        Processes the actions of the agent.

        Args:
            actions (List[Any]): The list of actions to process.
        """
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """
        Processes a tool call.

        Args:
            action (Any): The tool call action to process.
        """
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """
        Processes the observations.

        Args:
            observations (List[Any]): The list of observations to process.
        """
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """
        Processes the final result.

        Args:
            result (str): The final result to process.
        """
        self.callbacks.result_callback(result)
        self.output = result



def fetch_google_news(topic="technology", num_articles=5, country="CA"):
    """
    Fetches, prints, and returns the latest Google News articles using RSS for a specific country.

    :param topic: The topic to search news for (default: "technology").
    :param num_articles: Number of articles to display (default: 5).
    :param country: Country code (default: "CA" for Canada).
    :return: List of dictionaries containing title, link, and summary.
    """
    encoded_topic = urllib.parse.quote(topic)  # Encode query properly
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-{country}&gl={country}&ceid={country}:en"

    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print("No news articles found. Try a different topic.")
        return []

    articles = []

    print(f"\nLatest Google News in {country.upper()} for '{topic}':\n")
    for idx, entry in enumerate(feed.entries[:num_articles], 1):
        print(f"{idx}. {entry.title}")
        print(f"   {entry.link}\n")

        articles.append(
            {"title": entry.title, "link": entry.link, "summary": entry.summary}
        )

    return articles


# If you want to process the returned articles
# for news in news_list:
#     print(f"Title: {news['title']}\nLink: {news['link']}\nSummary: {news['summary']}\n")


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
