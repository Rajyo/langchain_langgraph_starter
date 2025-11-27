from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)
parser = StrOutputParser()


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedbackfor the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)


structured_llm = llm.with_structured_output(EvaluationSchema)


class UPSCState(TypedDict):
    title: str
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


def create_essay(state: UPSCState):
    title = state["title"]
    prompt = PromptTemplate(
        template="Write an essay on {title}", input_variables=["title"]
    )
    chain = prompt | llm | parser
    essay = chain.invoke({"title": title})
    print("Essay", essay, "\n")
    return {"essay": essay}


def evaluate_language(state: UPSCState):
    essay = state["essay"]
    prompt = PromptTemplate(
        template="Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}",
        input_variables=["essay"],
    )
    chain = prompt | structured_llm
    output = chain.invoke({"essay", essay})
    print("Language Output", output, "\n")
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_thought(state: UPSCState):
    essay = state["essay"]
    prompt = PromptTemplate(
        template="Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {essay}",
        input_variables=["essay"],
    )
    chain = prompt | structured_llm
    output = chain.invoke({"essay", essay})
    print("Clarity Output", output, "\n")
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_analysis(state: UPSCState):
    essay = state["essay"]
    prompt = PromptTemplate(
        template="Evaluate the depth of analysis of thought of the following essay and provide a feedback and assign a score out of 10 \n {essay}",
        input_variables=["essay"],
    )
    chain = prompt | structured_llm
    output = chain.invoke({"essay", essay})
    print("Analysis Output", output, "\n")
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}


def final_evaluation(state: UPSCState):
    clarity_feedback = state["clarity_feedback"]
    language_feedback = state["language_feedback"]
    analysis_feedback = state["analysis_feedback"]
    prompt = PromptTemplate(
        template="Based on the following feedbacks create a summarized feedback \n language feedback - {language_feedback} \n depth of analysis feedback - {analysis_feedback} \n clarity of thought feedback - {clarity_feedback}",
        input_variables=["language_feedback", "analysis_feedback", "clarity_feedback"],
    )
    chain = prompt | llm | parser
    overall_feedback = chain.invoke(
        {
            "language_feedback": language_feedback,
            "analysis_feedback": analysis_feedback,
            "clarity_feedback": clarity_feedback,
        }
    )
    print("Overall Output", overall_feedback, "\n")
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


graph = StateGraph(UPSCState)

graph.add_node("create_essay", create_essay)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "create_essay")
graph.add_edge("create_essay", "evaluate_analysis")
graph.add_edge("create_essay", "evaluate_language")
graph.add_edge("create_essay", "evaluate_thought")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

initial_state = {"title": "Rise of AI in India"}
result = workflow.invoke(initial_state)
workflow.get_graph().print_ascii()
