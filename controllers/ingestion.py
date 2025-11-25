import json
from models.DocumentSource import DocumentSource
from models.IngestResponse import IngestResponse
from fastapi import File, UploadFile, Form
from typing import List
from utils.download_file_from_url import download_file_from_url
from utils.process_text_pipeline import process_text_pipeline
from utils.extract_text_from_bytes import extract_text_from_bytes

async def ingestion(
    test_id: str = Form(..., description="Unique identifier for the test/exam"),
    tenant_id: str = Form(..., description="Unique identifier for the tenant"),
    metadata: str = Form("{}", description="JSON string of global metadata"),
    documents_json: str = Form("[]", description="JSON string list of DocumentSource (URLs/Text)"),
    files: List[UploadFile] = File(None, description="List of binary files (PDF, DOCX, Images)")        
):
    
    """
    Ingest a batch of content into the Vector DB.
    Supports:
    - Binary Files (PDF, DOCX, Images) via multipart/form-data
    - URLs and Raw Text via 'documents_json' field
    """
    processed_count = 0
    errors = []
    
    # Parse global metadata
    try:
        global_metadata = json.loads(metadata)
        global_metadata['test_id'] = test_id
        global_metadata['tenant_id'] = tenant_id
    except json.JSONDecodeError:
        global_metadata = {"test_id": test_id, "tenant_id": tenant_id}

    # --- 1. Process JSON Sources (URLs / Text) ---
    try:
        parsed_documents_data = json.loads(documents_json)
        document_sources = [DocumentSource(**item) for item in parsed_documents_data]
        
        for doc in document_sources:
            try:
                extracted_text = ""
                
                if doc.text:
                    extracted_text = doc.text
                elif doc.url:
                    print(f"Downloading from URL: {doc.url}")
                    file_content = await download_file_from_url(doc.url)
                    file_ext = doc.url.split('.')[-1].lower() if not doc.file_type else doc.file_type
                    extracted_text = extract_text_from_bytes(file_content, file_ext)

                # Send to pipeline
                process_text_pipeline(extracted_text, global_metadata)
                processed_count += 1
                
            except Exception as e:
                errors.append(f"Failed to process source {doc.url or 'text'}: {str(e)}")

    except Exception as e:
        errors.append(f"Failed to parse document_json: {str(e)}")

    # --- 2. Process Binary Files (Multipart) ---
    if files:
        for file in files:
            try:
                print(f"Processing binary file: {file.filename}")
                content = await file.read()
                filename = file.filename.lower()
                
                file_ext = ""
                if filename.endswith(".pdf"):
                    file_ext = "pdf"
                elif filename.endswith(".docx") or filename.endswith(".doc"):
                    file_ext = "doc"
                elif filename.endswith(".txt"):
                    file_ext = "txt"
                else:
                     errors.append(f"Skipping unsupported file: {filename}")
                     continue
                
                # Extract
                extracted_text = extract_text_from_bytes(content, file_ext)
                
                # Pipeline
                process_text_pipeline(extracted_text, global_metadata)
                processed_count += 1
                
            except Exception as e:
                errors.append(f"Failed to process binary file {file.filename}: {str(e)}")
            finally:
                await file.close()

    return IngestResponse(
        status="completed_with_errors" if errors else "success",
        message=f"Ingestion processed {processed_count} items",
        processed_count=processed_count,
        errors=errors
    )