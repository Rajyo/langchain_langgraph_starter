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


class LLMState(TypedDict):
    question: str
    answer: str


def simple_llm(state: LLMState) -> LLMState:
    question = state["question"]
    prompt = PromptTemplate(
        template="Answer the following question {question}",
        input_variables=["question"],
    )
    chain = prompt | llm | parser
    answer = chain.invoke(question)
    state["answer"] = answer
    return state


graph = StateGraph(LLMState)
graph.add_node("simple_llm", simple_llm)

graph.add_edge(START, "simple_llm")
graph.add_edge("simple_llm", END)

workflow = graph.compile()

initial_state = {"question": "What is the capital of India"}
result = workflow.invoke(initial_state)
print(result)
