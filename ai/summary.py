import json
import os 
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

    

