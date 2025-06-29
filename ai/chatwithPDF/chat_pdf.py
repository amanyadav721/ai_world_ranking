
import os
import shutil
import hashlib
from typing import List,Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load environment variables from .env file
load_dotenv()


user_data_cache: Dict[str, Dict] = {}
FAISS_STORAGE_DIR = "faiss_indices"
os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)

def get_chat_model():
    llm = ChatGroq(
    model="llama3-70b-8192", # Using a recommended model, adjust if needed
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
    return llm

# Initialize a free, local embedding model from HuggingFace
# This model runs on your local machine.
google_api_token = os.getenv("GEMINI_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_token)

def get_hashed_email_path(user_email: str) -> str:
    """Generates a consistent and safe directory name for FAISS index from email."""
    hashed_email = hashlib.sha256(user_email.encode('utf-8')).hexdigest()
    return os.path.join(FAISS_STORAGE_DIR, hashed_email)

def load_or_create_vector_store(file_path: str, user_email_hash_path: str):
    """
    Loads FAISS from disk if exists, otherwise processes PDF, creates embeddings,
    and stores them in FAISS, then saves to disk.
    """
    if os.path.exists(user_email_hash_path) and os.path.isdir(user_email_hash_path):
        print(f"Loading existing vector store for {user_email_hash_path}...")
        try:
            vector_store = FAISS.load_local(user_email_hash_path, embedding_model, allow_dangerous_deserialization=True)
            print("Existing vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Recreating...")
            shutil.rmtree(user_email_hash_path) # Clean up corrupted store

    print(f"Creating new vector store for {user_email_hash_path}... This may take a moment.")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(texts, embedding_model)
    vector_store.save_local(user_email_hash_path)
    print("New vector store created and saved successfully.")
    return vector_store