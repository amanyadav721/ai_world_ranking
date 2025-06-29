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

class LoanRepayment(BaseModel):
    type: str
    amount: float

class FinancialAnalysisRequest(BaseModel):
    user_profile: dict
    income: dict
    expenses: dict
    savings_and_investments: dict
    debts: dict
    goals: dict
    problems_or_stuck: list

class MonthlyCashflow(BaseModel):
    total_income: float
    total_expenses: float
    available_to_invest_or_save: float

class AnalysisSection(BaseModel):
    savings_rate: str
    emergency_fund_status: str
    debt_to_income_ratio: str
    investment_diversification: str
    loan_burden: str

class ProblemResolution(BaseModel):
    user_problem: str
    response: str

class FinancialAnalysisResponse(BaseModel):
    status: str
    financial_health_score: int
    rating: str
    monthly_cashflow: MonthlyCashflow
    analysis: AnalysisSection
    recommendations: list
    problem_resolution: list
    ai_in_depth_analysis: str = None