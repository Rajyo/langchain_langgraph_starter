from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
import os

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

text_loader = TextLoader("./10_generative_rag_text_file.txt")
docs = text_loader.load()
# print(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=125, chunk_overlap=25)
chunks = splitter.split_documents(docs)
# print(chunks, "\n", chunks.__len__())

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_GEMINI_API_KEY
    ),
    persist_directory="./chroma_db",
)

retreiver = vector_store.as_retriever(search_kwargs={"k": 3})
# print(retreiver)


template = """Answer the question based only on this context:

{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_messages([("system", template)])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

rag_chain = (
    ({"context": retreiver, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "Tell me about the key areas of impact of AI"})
print("Result", result)
