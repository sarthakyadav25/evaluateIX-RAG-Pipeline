from pydantic import BaseModel, Field
from typing import Optional, Dict


class RetrieveRequest(BaseModel):
    """
    Payload for retrieving context (from Node.js).
    """
    question: str = Field(..., description="The question asked by the bot")
    query: str = Field(..., description="The candidate's chat message or query")
    filters: Dict[str, str] = Field(..., description="Must include test_id to filter scope")
    top_k: int = Field(3, description="Number of relevant chunks to retrieve")