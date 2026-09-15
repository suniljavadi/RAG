import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import deque

# Load environment variables
load_dotenv()

try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    api_key = None
api_key = api_key or os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Used only for this Streamlit session and never written to disk.",
    )
    if not api_key:
        st.info("Add a key in .streamlit/secrets.toml, set OPENAI_API_KEY, or paste one here to continue.")
        st.stop()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, api_key=api_key)

import os

# Load vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

index_path = "faiss_index"

if os.path.exists(index_path):
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    example_path = os.path.join(os.path.dirname(__file__), "example.txt")
    if not os.path.exists(example_path):
        st.error("FAISS index and example.txt are both missing.")
        st.stop()
    documents = TextLoader(example_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.info("Using an in-memory index built from example.txt. Build faiss_index for a persisted index.")



# Set up conversational memory
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=10)

# Template for conversational RAG
conversational_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History:
{history}

Context:
{context}

Question: {question}

Helpful Answer:"""

conversational_prompt = PromptTemplate.from_template(conversational_template)

conversational_chain = (
    {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: "\n".join(st.session_state.history)}
    | conversational_prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("Conversational RAG System")
st.write("Ask questions based on the loaded documents.")

# Display chat history
for msg in st.session_state.history:
    if msg.startswith("Human: "):
        st.write(f"**You:** {msg[7:]}")
    elif msg.startswith("Assistant: "):
        st.write(f"**Bot:** {msg[11:]}")

# Input for new question
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        # Get response
        result = conversational_chain.invoke(question)
        
        # Update history
        st.session_state.history.append(f"Human: {question}")
        st.session_state.history.append(f"Assistant: {result}")
        
        # Display new response
        st.write(f"**You:** {question}")
        st.write(f"**Bot:** {result}")
        
        # Rerun to update display
        st.rerun()
    else:
        st.warning("Please enter a question.")

# Button to clear history
if st.button("Clear History"):
    st.session_state.history.clear()
    st.rerun()
