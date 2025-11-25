from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import List

class IngestResponse(BaseModel):
    status: str
    message: str
    processed_count: int
    errors: List[str] = []