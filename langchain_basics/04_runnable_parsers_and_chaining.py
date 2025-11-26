from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

# ------------------------------------------------------
#               Simple Chaining
# ------------------------------------------------------
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | llm | parser


result = chain.invoke({"topic": "cricket"})
print(result)


chain.get_graph().print_ascii()


# ------------------------------------------------------
#               Sequential Chaining
# ------------------------------------------------------
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

# chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)
chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({"topic": "Cricket"})
print(result)

chain.get_graph().print_ascii()


# ------------------------------------------------------
#               Parallel Chaining
# ------------------------------------------------------
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "topic": RunnablePassthrough(),
        "notes": prompt1 | llm | parser,
        "quiz": prompt2 | llm | parser,
    }
)

merge_chain = prompt3 | llm | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"topic": "Cricket"})
print(result)

chain.get_graph().print_ascii()


# ------------------------------------------------------
#               Conditional Chaining
# ------------------------------------------------------
class Feedback(BaseModel):
    sentiment: str = Field(description="Give the feedback of the sentiment")


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()},
)

classifier_chain = prompt1 | llm | pydantic_parser

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
)

parser = StrOutputParser()

positveLambda = RunnableLambda(
    lambda x: x.sentiment == "positive", prompt2 | llm | parser
)
negativeLambds = RunnableLambda(
    lambda x: x.sentiment == "negative", prompt3 | llm | parser
)

branch_chain = RunnableBranch(
    positveLambda,
    negativeLambds,
    RunnableLambda(lambda x: "could not find sentiment"),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This is a beautiful phone"})
print(result)

chain.get_graph().print_ascii()
