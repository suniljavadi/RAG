import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Check API key
api_key="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"
#api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env file. Please set it.")
    exit(1)

# Check document path
doc_path = "data/docs.txt"

if not os.path.exists(doc_path):
    print(f"❌ Document file not found: {doc_path}")
    print("Please create data/docs.txt and add some content.")
    exit(1)

# Load documents
print("📄 Loading documents...")
loader = TextLoader(doc_path)
documents = loader.load()

# Split documents
print("✂️ Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# Embeddings
print("🔎 Creating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector index path
index_path = "faiss_index"

# Avoid rebuilding index
if os.path.exists(index_path):
    print("⚠️ FAISS index already exists. Skipping creation.")
else:
    print("📦 Building FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save index
    vectorstore.save_local(index_path)

    print("✅ FAISS index created successfully!")

print("🎉 Indexing process completed.")