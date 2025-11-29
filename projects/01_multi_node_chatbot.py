#########################################################
#                       Multi Node
#########################################################

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

memory = InMemorySaver()


class ChatModel(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_node: Literal["greet", "chat", "farewell"]


def greet(state: ChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Bot, a friendly assistant with dry humor.
        This is your GREETING message. Be welcoming, brief, and engaging.
        End by asking how you can help. Transition to chat mode.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(state)
    return {"messages": [response], "next_node": "chat"}


def chat(state: ChatModel):
    human_input = input("You: ").strip().lower()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Bot, a helpful assistant with dry humor.
        You are maximally truthful and engaging.
        Keep responses concise (2-3 sentences max).
        If user wants to end conversation, transition to farewell.
        Look for keywords: bye, goodbye, exit, quit, done, farewell""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("human", human_input),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(state)

    ending_keywords = ["bye", "goodbye", "exit", "quit", "done", "farewell", "end"]

    if any(keywords in human_input for keywords in ending_keywords):
        return {"messages": [response], "next_node": "farewell"}
    return {"messages": [response], "next_node": "chat"}


def farewell(state: ChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Bot. This is your FAREWELL message.
        Be warm, brief, and memorable. Thank the user for chatting.
        Include a touch of dry humor. End the conversation gracefully.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(state)
    return {"messages": [response], "next_node": END}


graph = StateGraph(ChatModel)

graph.add_node("greet", greet)
graph.add_node("chat", chat)
graph.add_node("farewell", farewell)

graph.add_edge(START, "greet")
graph.add_edge("greet", "chat")
graph.add_conditional_edges(
    "chat",
    lambda state: state.get("next_node", "chat"),
    {"chat": "chat", "farewell": "farewell"},
)
graph.add_edge("farewell", END)

workflow = graph.compile(checkpointer=memory)
print("================================================================")
print("                      Multi Node Chatbot                        ")
print("================================================================ \n")
print("Hi my name is Bot, whats your name \n")
while True:
    config = {"configurable": {"thread_id": "123"}}
    user_input = input("You: ").strip()

    if user_input in {"bye", "goodbye", "exit", "quit", "done", "farewell", "end"}:
        print("👋 Conversation ended. Thanks for chatting!")
        break

    human_message = HumanMessage(content=user_input)

    for chunk in workflow.stream(
        {"messages": human_message, "next_node": "greet"}, config
    ):
        if "greet" in chunk:
            print("Bot:", chunk["greet"]["messages"][0].content, "\n")
        elif "chat" in chunk:
            print("Bot:", chunk["chat"]["messages"][0].content, "\n")
        else:
            print("Bot:", chunk["farewell"]["messages"][0].content, "\n")
            print("👋 Conversation ended. Thanks for chatting!", "\n")
            break
    break


#########################################################
#                       Single Node
#########################################################


# from langgraph.graph import START, END, StateGraph
# from typing import TypedDict, Annotated
# from langgraph.graph.message import BaseMessage, add_messages
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain.messages import HumanMessage
# from langgraph.checkpoint.memory import InMemorySaver
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# load_dotenv()
# import os

# GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

# memory = InMemorySaver()


# class ChatMessages(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant named Bot.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )


# def chatbot(state: ChatMessages):
#     # full_prompt = prompt.invoke(state)
#     # response = llm.invoke(full_prompt)
#     chain = prompt | llm
#     response = chain.invoke(state)
#     return {"messages": [response]}


# graph = StateGraph(ChatMessages)
# graph.add_node("chatbot", chatbot)

# graph.add_edge(START, "chatbot")
# graph.add_edge("chatbot", END)

# workflow = graph.compile(checkpointer=memory)

# while True:
#     config = {"configurable": {"thread_id": "123"}}
#     user_input = input("You: ")
#     if user_input.lower() in {"end", "quit", "exit"}:
#         print("Bot: Goodbye")
#         break
#     human_message = HumanMessage(content=user_input)

#     for chunk in workflow.stream({"messages": [human_message]}, config=config):
#         print("Bot:", chunk["chatbot"]["messages"][0].content, "\n")


#########################################################
#                       Single Node
#########################################################


# from langgraph.graph import START, END, StateGraph
# from typing import TypedDict, Annotated
# from langgraph.graph.message import BaseMessage, add_messages
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain.messages import AIMessage, HumanMessage

# load_dotenv()
# import os

# GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)


# class ChatMessages(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]


# def chatbot(state: ChatMessages):
#     response = llm.invoke(conversation_history["messages"])
#     conversation_history["messages"].append(response)
#     return {"messages": [response]}


# graph = StateGraph(ChatMessages)
# graph.add_node("chatbot", chatbot)

# graph.add_edge(START, "chatbot")
# graph.add_edge("chatbot", END)

# workflow = graph.compile()

# conversation_history = {"messages": []}

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in {"end", "quit", "exit"}:
#         print(result)
#         print("Bot: Goodbye")
#         break
#     human_message = HumanMessage(content=user_input)
#     conversation_history["messages"].append(human_message)
#     result = workflow.invoke({"messages": [human_message]})
#     ai_response = result["messages"][-1].content
#     print(f"Bot: {ai_response}")
# print(conversation_history)
