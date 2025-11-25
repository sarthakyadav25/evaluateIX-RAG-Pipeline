from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Optional

class DocumentSource(BaseModel):
    """
    Represents a single document to be ingested via URL or Text.
    """
    url: Optional[str] = Field(None, description="S3/Storage URL for a PDF, DOCX, or TXT file")
    text: Optional[str] = Field(None, description="Direct raw text content")
    file_type: Optional[str] = Field(None, description="Explicit file type (pdf, docx) if URL doesn't have extension")