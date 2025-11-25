from pydantic import BaseModel
from typing import List


class QuestionItem(BaseModel):
    question_no: int
    content: str

class QuestionGenerationResponse(BaseModel):
    questions: List[QuestionItem]