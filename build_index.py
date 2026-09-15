import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not found in .env file. Please set it.")
    raise SystemExit(1)

doc_path = "data/docs.txt"

if not os.path.exists(doc_path):
    print(f"Document file not found: {doc_path}")
    print("Please create data/docs.txt and add some content.")
    raise SystemExit(1)

print("Loading documents...")
loader = TextLoader(doc_path)
documents = loader.load()

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

print("Creating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_path = "faiss_index"

if os.path.exists(index_path):
    print("FAISS index already exists. Skipping creation.")
else:
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print("FAISS index created successfully!")

print("Indexing process completed.")
