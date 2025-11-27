from langgraph.graph import START, END, StateGraph
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

llm = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)
parser = StrOutputParser()


class BlogState(TypedDict):
    title: str
    outline: str
    content: str


def create_outline(state: BlogState) -> BlogState:
    title = state["title"]
    prompt = PromptTemplate(
        template="Generate a detailed outline for a blog on the topic - {title}",
        input_variables=["title"],
    )
    chain = prompt | llm | parser
    outline = chain.invoke(title)
    state["outline"] = outline
    return state


def create_blog(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]
    prompt = PromptTemplate(
        template="Write a detailed blog on the title - {title} using the follwing outline \n {outline}",
        input_variables=["title", "outline"],
    )
    chain = prompt | llm | parser
    content = chain.invoke({"title": title, "outline": outline})
    state["content"] = content
    return state


graph = StateGraph(BlogState)

graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

initial_state = {"title": "Rise of AI in India"}
result = workflow.invoke(initial_state)
print(result)
