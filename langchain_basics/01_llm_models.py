from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# ------------------------------------------------------
#               Google Generative AI
# ------------------------------------------------------
llm = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

# ------------------------------------------------------
#               Normal Response
# ------------------------------------------------------
response = llm.invoke("explain quantum computing simply")
print("Response", response)

# ------------------------------------------------------
#               Streaming Response
# ------------------------------------------------------
for chunk in llm.stream("explain quantum computing simply"):
    print(chunk, end="", flush=True)


# ------------------------------------------------------
#               Chat Google Generative AI
# ------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="explain quantum computing simply"),
]

# ------------------------------------------------------
#               Normal Response
# ------------------------------------------------------
response = llm.invoke(messages)
print("Response", response)

# ------------------------------------------------------
#               Appending Messages Response
# ------------------------------------------------------
response = llm.invoke(messages)
messages.append(AIMessage(content=response.content))
print(messages)

# ------------------------------------------------------
#               Streaming Response
# ------------------------------------------------------
for chunk in llm.stream("explain quantum computing simply"):
    print(chunk.content, end="", flush=True)
