from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

# Load and process documents
documents = TextLoader("example.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(texts, embeddings)

# Set up RAG chain
retriever = vectorstore.as_retriever()
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Interactive Q&A
while True:
    question = input("Enter your question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    result = rag_chain.invoke(question)
    print(f"Answer: {result}\n")