from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)


# ------------------------------------------------------
#               Pydantic Structured Output
# ------------------------------------------------------
class Character(BaseModel):
    name: str = Field(..., description="Full name")
    age: int = Field(..., description="Age in years")
    skills: List[str] = Field(..., description="List of skills")


structured_llm = llm.with_structured_output(Character)

result = structured_llm.invoke("Tell me about Albus Dumbledore")
print(result)
