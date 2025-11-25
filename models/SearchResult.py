from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]