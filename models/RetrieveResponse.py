from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from models.SearchResult import SearchResult


class RetrieveResponse(BaseModel):
    results: List[SearchResult]
    answer: Dict[str,Any] = Field(..., description="The generated answer from the LLM")