from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

# ------------------------------------------------------
#               Prompt Template
# ------------------------------------------------------
prompt_template = PromptTemplate(
    template="Greet this person in  languages. The name of the person is {name}",
    input_variables=["name"],
)

prompt = prompt_template.invoke({"name": "Phil"})

result = llm.invoke(prompt)
print(result.content)


# ------------------------------------------------------
#               Chat Prompt Template
# ------------------------------------------------------
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert"),
        ("human", "Explain {topic} in simple terms"),
    ]
)

prompt = chat_prompt_template.invoke({"domain": "Cricket", "topic": "Dusra"})

result = llm.invoke(prompt)
print(result.content)


# ------------------------------------------------------
#               Message Placeholder
# ------------------------------------------------------
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support agent"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

chat_history = []

with open("02_chat_history.txt") as file:
    chat_history.extend(file.readlines())

print(chat_history)

prompt = chat_prompt_template.invoke(
    {"history": chat_history, "query": "Where is my refund"}
)

result = llm.invoke(prompt)
print(result.content)


# ------------------------------------------------------
#               Simple Chatbot
# ------------------------------------------------------
chatbot_history = [SystemMessage(content="You are a helpful AI Assistant")]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "exit":
        break

    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI ", result.content)

print(chat_history)
