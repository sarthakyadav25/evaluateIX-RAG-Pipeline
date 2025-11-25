import dotenv
import uvicorn
import json
import re
import io
import httpx
import uuid
import os
from contextlib import asynccontextmanager
from pypdf import PdfReader
import docx
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from controllers.ingestion import ingestion
from controllers.retrieval import retrieval
from controllers.question_generation import question_generation
from security.auth import verify_token
from fastapi import Depends
from models.DocumentSource import DocumentSource
from models.IngestResponse import IngestResponse
from models.RetrieveRequest import RetrieveRequest
from models.RetrieveResponse import RetrieveResponse
from models.SearchResult import SearchResult
from models.QuestionGenerationResponse import QuestionGenerationResponse
from models.QuestionGenerationRequest import QuestionGenerationRequest
from utils.parse_markdown_json import parse_markdown_json
from utils.download_file_from_url import download_file_from_url
from utils.extract_text_from_bytes import extract_text_from_bytes
from utils.process_text_pipeline import process_text_pipeline
from utils.rag_initialization import embedding_model,GEMINI_API_KEY,collection
from utils.rag_initialization import rag_initialization

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs BEFORE the server starts accepting requests.
    It is the perfect place to load ML models and DB connections.
    """
    print("startup: Triggering RAG Initialization...")
    try:
        # Run your initialization logic here
        rag_initialization() 
        print("startup: RAG Initialization Complete.")
    except Exception as e:
        print(f"startup: CRITICAL ERROR during initialization: {e}")
        # You might want to raise the error to stop the server if models fail
        raise e
    
    yield # The server runs and handles requests here
    
    # (Optional) Code here runs when the server shuts down
    print("shutdown: Cleaning up resources...")


# --- Configuration ---
app = FastAPI(
    title="RAG Microservice",
    description="Vector ingestion and retrieval service for AI Bot Platform",
    version="1.3.0",
    lifespan=lifespan,
)



# --- API Endpoints ---

@app.post("/ingest", response_model=IngestResponse, status_code=202, dependencies=[Depends(verify_token)])
async def ingest_content(
    test_id: str = Form(..., description="Unique identifier for the test/exam"),
    tenant_id: str = Form(..., description="Unique identifier for the tenant"),
    metadata: str = Form("{}", description="JSON string of global metadata"),
    documents_json: str = Form("[]", description="JSON string list of DocumentSource (URLs/Text)"),
    files: List[UploadFile] = File(None, description="List of binary files (PDF, DOCX, Images)")
):
    return await ingestion(test_id, tenant_id, metadata, documents_json, files)
 

@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(verify_token)])
async def retrieve_context(payload: RetrieveRequest):
    return await retrieval(payload)

@app.post("/generate-questions", response_model=QuestionGenerationResponse, dependencies=[Depends(verify_token)])
async def generate_questions(payload: QuestionGenerationRequest):
    return await question_generation(payload)
        


@app.get("/health")
async def health_check():
    """Health check for k8s/monitoring"""
    return {"status": "operational", "service": "rag-pipeline"}

if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000, reload=True)