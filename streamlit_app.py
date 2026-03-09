import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import deque

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file. Please set it.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

# Load vectorstore
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}. Please ensure faiss_index exists.")
    st.stop()

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
    