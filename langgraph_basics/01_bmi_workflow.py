from langgraph.graph import START, END, StateGraph
from typing import TypedDict


class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: float


def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]

    bmi = weight / (height**2)

    state["bmi"] = round(bmi, 2)
    return state


def label_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]

    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state


graph = StateGraph(BMIState)

graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmi", label_bmi)

graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")
graph.add_edge("label_bmi", END)

workflow = graph.compile()

initial_state = {"weight": 66, "height": 1.73}
final_state = workflow.invoke(initial_state)
print(final_state)
