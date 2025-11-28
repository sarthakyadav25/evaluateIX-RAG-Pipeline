from typing import List
from pydantic import UUID4, BaseModel, Field



class QuestionGenerationRequest(BaseModel):
    test_id: UUID4 = Field(..., description="ID of the test to generate questions for")
    num_questions: int = Field(5, description="Number of questions to generate")
    difficulty: str = Field("medium", description="Difficulty level: easy, medium, hard")
    already_has: List[str] = Field(..., description="List of already existing questions")