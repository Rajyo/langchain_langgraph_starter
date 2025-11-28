from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_weather(location: str):
    """Get current weather for a city."""
    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={location}"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_weather]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()
chatbot.get_graph().print_ascii()

result = chatbot.invoke(
    {"messages": [HumanMessage(content="What is the current weather in Mumbai")]}
)
print(result["messages"][-1].content, "\n")

result = chatbot.invoke(
    {"messages": [HumanMessage(content="What is the stock price of apple")]}
)
print(result["messages"][-1].content, "\n")
