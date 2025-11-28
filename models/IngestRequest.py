from fastapi import File, Form, UploadFile
from pydantic import UUID4, BaseModel
from typing import List


class IngestRequest(BaseModel):
    test_id: UUID4 = Form(..., description="Unique identifier for the test/exam"),
    tenant_id: UUID4 = Form(..., description="Unique identifier for the tenant"),
    metadata: str = Form("{}", description="JSON string of global metadata"),
    documents_json: str = Form("[]", description="JSON string list of DocumentSource (URLs/Text)"),
    files: List[UploadFile] = File(None, description="List of binary files (PDF, DOCX, Images)")   