import chromadb
import os
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


# --- RAG Initialization ---

# 1. Define globals as None initially
embedding_model = None
chroma_client = None
collection = None
GEMINI_API_KEY = None

def rag_initialization():
    """Initializes global variables"""
    global embedding_model, chroma_client, collection, GEMINI_API_KEY
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="rag_knowledge_base")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)