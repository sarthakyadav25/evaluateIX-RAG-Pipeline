import json
from pydantic import UUID4
import uvicorn
from loguru import logger
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from typing import List
from controllers.ingestion import ingestion
from controllers.retrieval import retrieval
from controllers.question_generation import question_generation
from security.auth import verify_token
from fastapi import Depends
from models.IngestResponse import IngestResponse
from models.IngestRequest import IngestRequest
from models.RetrieveRequest import RetrieveRequest
from models.RetrieveResponse import RetrieveResponse
from models.QuestionGenerationResponse import QuestionGenerationResponse
from models.QuestionGenerationRequest import QuestionGenerationRequest
from utils.rag_initialization import rag_initialization
from utils.redis_init import redis_init

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs BEFORE the server starts accepting requests.
    It is the perfect place to load ML models and DB connections.
    """
    logger.info("startup: Triggering RAG Initialization...")
    try:
        # Run your initialization logic here
        rag_initialization()
        logger.info("startup: RAG Initialization Complete.")
    except Exception as e:
        logger.critical(f"startup: CRITICAL ERROR during initialization: {e}")
        # You might want to raise the error to stop the server if models fail
        raise e
    
    try:
        #initializing redis
        redis_init()
        logger.info("Redis Initialized...")
    except Exception as e:
        logger.critical(f"startup: CRITICAL ERROR during redist initialization: {e}")
        raise e
    
    yield # The server runs and handles requests here
    
    # (Optional) Code here runs when the server shuts down
    logger.info("shutdown: Cleaning up resources...")


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
    test_id: UUID4 = Form(..., description="Unique identifier for the test/exam"),
    tenant_id: UUID4 = Form(..., description="Unique identifier for the tenant"),
    metadata: str = Form("{}", description="JSON string of global metadata"),
    documents_json: str = Form("[]", description="JSON string list of DocumentSource (URLs/Text)"),
    files: List[UploadFile] = File(None, description="List of binary files (PDF, DOCX, Images)")   
):
    
    try:
        json.loads(metadata)
        json.loads(documents_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON format: {str(e)}")

    # 3. Create the Payload Object
    payload = IngestRequest(
        test_id=test_id,
        tenant_id=tenant_id,
        metadata=metadata,
        documents_json=documents_json,
        files=files
    )
    return await ingestion(payload)
 

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