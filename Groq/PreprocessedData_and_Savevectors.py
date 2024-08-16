import os
import faiss
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai_api_key = os.environ["OPENAI_API_KEY"]

# Step 1: Load documents
print("Loading documents...")
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

# Step 2: Split the documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
print(f"Split into {len(final_documents)} chunks.")

# Step 3: Generate embeddings using OpenAI
print("Generating embeddings using OpenAI...")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectors = FAISS.from_documents(final_documents, embeddings)
print("Embeddings generated.")

# Step 4: Save the vector store using faiss
print("Saving vector store...")
faiss.write_index(vectors.index, "vector_store.index")
print("Vector store saved.")
