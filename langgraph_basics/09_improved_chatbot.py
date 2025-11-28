from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)


SYSTEM_PROMPT = (
    "You are a friendly, witty assistant named Bot. "
    "You love dry humor and always stay maximally truthful. "
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def chatbot(state: State):
    full_prompt = prompt.invoke(state)
    response = llm.invoke(full_prompt)
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


memory = MemorySaver()
app = builder.compile(checkpointer=memory)


def run_chat():
    print("Hi! I'm your stateful assistant. (type 'quit' to exit)\n")
    config = {"configurable": {"thread_id": "user_42"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Bot: Catch you later!")
            break

        print("Bot: ", end="", flush=True)
        for chunk in app.stream(
            {"messages": [("human", user_input)]},
            config,
            stream_mode="values",
        ):
            msg = chunk["messages"][-1]
            if msg.type == "ai":
                print(msg.content, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    run_chat()


# from typing import Annotated
# from typing_extensions import TypedDict
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages, BaseMessage
# from langgraph.checkpoint.memory import MemorySaver
# from dotenv import load_dotenv

# load_dotenv()
# import os

# GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


# class State(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]


# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

# graph_builder = StateGraph(State)


# def chatbot(state: State):
#     response = llm.invoke(state["messages"])
#     return {"messages": [response]}


# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)


# def chat():
#     print("Hi! I'm your stateful assistant. (type 'quit' to exit)\n")
#     thread_id = {"configurable": {"thread_id": "user_123"}}

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in {"exit", "quit"}:
#             print("Goodbye!")
#             break

#         print("Bot: ", end="", flush=True)
#         for chunk in graph.stream(
#             {"messages": [("human", user_input)]},
#             thread_id,
#             stream_mode="values",
#         ):

#             last_message = chunk["messages"][-1]
#             if last_message.type == "ai":
#                 print(last_message.content, end="", flush=True)
#         print("\n")


# if __name__ == "__main__":
#     chat()
