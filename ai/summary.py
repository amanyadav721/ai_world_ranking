import json
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=30,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def sanitize_json_string(s: str) -> str:
    """Remove control characters that break JSON parsing."""
    return re.sub(r"[\x00-\x1F\x7F]", "", s)

def resume_analysis(data, job_title: str, job_description: str) -> dict:
    """
    Analyzes the given resume using the Groq LLM.

    Args:
        data (str): Cleaned resume text
        job_title (str): Job title
        job_description (str): JD

    Returns:
        dict: Parsed JSON response
    """
    system_prompt = f"""
    You are Ayodhan, an intelligent and supportive AI assistant specialized in resume optimization for global job markets.
    Your primary role is to help users craft highly effective, ATS (Applicant Tracking System)-friendly resumes with a target score of at least 95% compatibility for the job description provided.
    User will provide you resume text, job title, and job description.
    You will analyze the resume against the job description and provide a detailed ATS score evaluation, improvement suggestions, and formatting guidance.
    Job Title: {job_title}
    Job Description: {job_description}

    Note: IF job description and job title are not provided, you should analyze the resume based on general best practices for ATS optimization.

    Output Format:
    {{
        "user_name": string,
        "user_email": string,
        "ats_score": float,
        "analysis": string,
        "good_points": list,
        "suggestions": list of dicts,
        "redundant_sections": list of strings,
        "keyword_match": {{
            "matched_keywords": list,
            "missing_keywords": list,
            "match_percentage": float
        }},
        "readability_score": {{
            "grade_level": float,
            "readability": string,
            "avg_sentence_length": float
        }}
    }}

    Strict Guidelines:
    1. No markdown or formatting characters (e.g. ```).
    2. No explanation outside the JSON.
    3. Escape all special characters.
    4. Response must start with {{ and end with }}.
    5. Always return valid JSON structure.
    """

    messages = [
        ("system", system_prompt),
        ("human", f"This is the scraped text from user's resume:\n{data}")
    ]

    ai_msg = llm.invoke(messages)
    raw_output = ai_msg.content

    cleaned_output = sanitize_json_string(raw_output)

    try:
        response = json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        print("Model response:", repr(cleaned_output))
        raise ValueError(f"Failed to parse JSON from LLM: {str(e)}")

    return response



