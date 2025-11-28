import chromadb
import os
from chromadb.config import Settings

import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
import numpy as np
from loguru import logger

load_dotenv()


# --- RAG Initialization ---

# 1. Define globals as None initially
embedding_model = None
chroma_client = None
collection = None
GEMINI_API_KEY = None

# 1. Define the Adapter Class
class GoogleEmbeddingAdapter:
    def __init__(self, google_chroma_func):
        self.google_func = google_chroma_func

    def encode(self, documents, **kwargs):
        # 1. Check if input is a single string or a list
        is_single_string = isinstance(documents, str)
        
        # 2. Google API always expects a list
        if is_single_string:
            documents = [documents]

        # 3. Get embeddings (Google always returns a list of lists)
        embeddings = self.google_func(documents)

        # 4. CRITICAL FIX: If input was a single string, return a 1D array.
        # This matches SentenceTransformer behavior exactly.
        if is_single_string:
            return np.array(embeddings[0])
        
        # Otherwise return the 2D array
        return np.array(embeddings)

def rag_initialization():
    """Initializes global variables"""
    global embedding_model, chroma_client, collection, GEMINI_API_KEY

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    logger.info("Loading embedding model...")
    embedding_model = GoogleEmbeddingAdapter(embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name="models/text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT" # Optimizes embeddings for storage/retrieval
    ))

    CHROMA_DB_CLOUD = os.getenv("CHROMA_DB_CLOUD")
    if CHROMA_DB_CLOUD:
        logger.info("Initializing ChromaDB...")
        # chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_client = chromadb.CloudClient(
            api_key=CHROMA_DB_CLOUD,
            tenant='c099f9b2-faf5-445f-8e03-12e11fa8b460',
            database='evaluateIX-RAG-Pipeline ')
        collection = chroma_client.get_or_create_collection(name="rag_knowledge_base_v1")
    else:
        logger.warning("ChromaDB Cloud API KEY not found...")