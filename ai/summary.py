import json
import os 
import io
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=30,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def summarize_text(data:dict) -> str:
    """
    Summarizes the given text using the Groq LLM.
    
    Args:
        text (str): The text to summarize.
    
    Returns:
        str: The summarized text.

    """

    user_data = data

    system_prompt = f"""
    You are Ayodhan , a helpful AI assistant.
    You taks is to understand the scrapped content and provide a summary of the content.
    The summary should be concise and informative.
    The summary should contains the all important points from the content so we can use it later for analysis.
    You should not include any personal opinions or irrelevant information.

    Output Format:
    {{
        "user_name":,
        "user_email": string,
        "summary": string,
        "skills": List[string],
        
    }}
    Strict Guideliness:
                    1. No text outside the JSON object.

                    2. No markdown or code blocks in the JSON object.

                    3. Must be valid JSON open and close braces.

                    4. Structure must be exactly as specified, with no extra fields or changes.

                    5. Do not add ''' or '''json to the JSON object before or after the JSON object.

                    7. You reponse should start with open brace and end with close brace.
    """

    
    messages = [
    ("system", system_prompt),
    ("human", json.dumps(user_data)),  # Convert list/dict to a JSON string
  ]

    ai_msg = llm.invoke(messages)
    repsonse = json.loads(ai_msg.content)
    return repsonse

    


def resume_analysis(data:str,job_title:str, job_description:str) -> json:
    """
    Analyzes the given resume using the Groq LLM.

    Args:
        text (str): The text to summarize.
    
    Returns:
        json: The analysis results.

    """

    user_data = data

    system_prompt = f"""
    You are Ayodhan, an intelligent and supportive AI assistant specialized in resume optimization for global job markets.
    Your primary role is to help users craft highly effective, ATS (Applicant Tracking System)-friendly resumes with a target score of at least 95% compatibility for the job description provided.
    User will provide you resume text, job title, and job description.
    You will analyze the resume against the job description and provide a detailed ATS score evaluation, improvement suggestions, and formatting guidance.
    Job Title: {job_title}
    Job Description: {job_description}

    Note: IF job description and job title are not provided, you should analyze the resume based on general best practices for ATS optimization.
    Tasks:
    ATS Score Evaluation

    Analyze the user's resume in comparison with the provided job description.

    Assign a precise ATS score ranging from 0 to 100 based on alignment with key ATS parameters including:

    Keyword relevance

    Role-specific skills

    Formatting

    Job title matching

    Education and certifications

    Experience relevance

    Detailed Improvement Suggestions

    Provide specific, actionable, and concise suggestions to improve the ATS score.

    Ensure that each suggestion includes a clear reason and, where applicable, example rewrites or templates.

    Suggestions must:

    Be tailored to the user's current resume and the job description.

    Avoid generic or repetitive advice.

    Clearly identify which section or text block the suggestion applies to (e.g., “Experience”, “Skills”, “Summary”).

    Include formatting recommendations such as structure, bullet points, sentence length, or removal of graphics/tables not readable by ATS.

    Formatting and Readability Guidance

    Ensure that the resume follows global best practices:

    Clean layout with standard fonts (Arial, Calibri, Times New Roman)

    Proper use of headings and bullet points

    No use of columns, graphics, or text boxes that ATS systems can’t read

    Consistent date formatting

    Clear job titles and chronological order of experience

    Output Format:
    {{
        "user_name": string,
        "user_email": string,
        "ats_score": float (0-100),
        "analysis": string,
        "good_points": List[string],
        "suggestions": List[dict],
        "redundant_sections": List[str] example:[
            "Objective"  // if outdated or too generic
        ],
        "keyword_match": {{
        "matched_keywords":List[str] 
        "missing_keywords":List[str] 
        "match_percentage": float 
    }},

    "readability_score": {{
        "grade_level": float 
        "readability": str 
        "avg_sentence_length": float 
     }},
        

    }}
    Strict Guideliness:
                    1. No text outside the JSON object.

                    2. No markdown or code blocks in the JSON object.

                    3. Must be valid JSON open and close braces.

                    4. Structure must be exactly as specified, with no extra fields or changes.

                    5. Do not add ''' or '''json to the JSON object before or after the JSON object.

                    7. You reponse should start with open brace and end with close brace.
    """

    
    messages = [
    ("system", system_prompt),
    ("human", f"This scrapped text from user's resume{user_data}"),  # Convert list/dict to a JSON string
  ]

    ai_msg = llm.invoke(messages)
    repsonse = json.loads(ai_msg.content)
    return repsonse




