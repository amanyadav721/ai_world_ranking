from pydantic import BaseModel, Field
from typing import Optional,List,Dict

class QuestionData(BaseModel):
    practice_type: str = Field(..., description="The type of practice question")
    difficulty: str = Field(..., description="The difficulty level of the question")
    topic: str = Field(..., description="The topic of the question")
    question_type: Optional[str] = Field(None, description="The type of question (e.g., 'mcq', 'text', 'coding')")
    number_of_questions: int = Field(..., description="The number of questions to generate")


class QuestionAnalyse(BaseModel):
    questionData: dict = Field(..., description="The question data containing practice type, difficulty, topic, and number of questions")
    questions: List[Dict] = Field(..., description="A list of generated questions with their answers")
    time_taken: Optional[float] = Field(None, description="Time taken to generate the questions in seconds")
    user_solution: Optional[str] = Field(None, description="User's solution to the question")


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    user_email: str
    query: str

class UploadResponse(BaseModel):
    status: str
    message: str

class ChatResponse(BaseModel):
    answer: str