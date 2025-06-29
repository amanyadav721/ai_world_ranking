import os
import json
from typing import Dict, Any
from Models.modelsv1 import FinancialAnalysisRequest, FinancialAnalysisResponse
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=30,
    max_retries=2,
    groq_api_key=groq_api_key,
)

def build_financial_prompt(data: Dict[str, Any]) -> str:
    """
    Build a detailed prompt for the Groq LLM to analyze the user's financial data and return a structured JSON response.
    """
    return f"""
You are a world-class financial advisor AI. Analyze the following user's financial data and return a JSON object with the following structure:
{{
  \"status\": \"success\",
  \"financial_health_score\": int, // 0-100
  \"rating\": string, // Excellent, Good, Adequate, Poor
  \"monthly_cashflow\": {{
    \"total_income\": float,
    \"total_expenses\": float,
    \"available_to_invest_or_save\": float
  }},
  \"analysis\": {{
    \"savings_rate\": string, // e.g. '12.8%'
    \"emergency_fund_status\": string, // e.g. 'Below recommended (25k vs 3 months expenses = 90k)'
    \"debt_to_income_ratio\": string, // e.g. '35.8%'
    \"investment_diversification\": string, // e.g. 'Low (100% gold)'
    \"loan_burden\": string // e.g. 'High EMI load, car + education = 36.5% of income'
  }},
  \"recommendations\": [string],
  \"problem_resolution\": [{{\"user_problem\": string, \"response\": string}}],
  \"ai_in_depth_analysis\": string // A detailed, personalized analysis and improvement plan
}}
Strictly follow this output format. Do not include any text outside the JSON. Here is the user's data:
 Strict Guidelines:
                    1. No text outside the JSON object.

                    2. No markdown or code blocks in the JSON object.

                    3. Must be valid JSON open and close braces.

                    4. Structure must be exactly as specified, with no extra fields or changes.

                    5. Do not add ''' or '''json to the JSON object before or after the JSON object.

                    7. You reponse should start with open brace and end with close brace.
{json.dumps(data)}
"""

def analyze_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_financial_prompt(data)
    ai_msg = llm.invoke([("system", prompt)])
    raw_output = ai_msg.content
    # Clean and parse JSON
    cleaned = raw_output.strip().replace("\n", " ").replace("\r", "")
    try:
        result = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Groq LLM output could not be parsed as JSON: {e}\nRaw: {cleaned}")
    return result 