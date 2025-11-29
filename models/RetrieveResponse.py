from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from models.SearchResult import SearchResult


class RetrieveResponse(BaseModel):
    results: List[SearchResult]
    answer: Dict[str,Any] = Field(..., description="The generated answer from the LLM")
    ai_score: float = Field(..., description="A score out of 100 to detect whether content is AI written")