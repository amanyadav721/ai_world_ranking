import json
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import re

def safe_json_parse(text: str):
    try:
        # Remove unescaped control characters (e.g., newlines inside strings)
        clean_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print("Failed to parse LLM output:\n", text)
        raise ValueError(f"Invalid JSON output: {e}")



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=30,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


    
def questionBuilderv1(data:dict) -> str:
    """
    Summarizes the given text using the Groq LLM.
    
    Args:
        text (str): The text to summarize.
    
    Returns:
        str: The summarized text.

    """


    user_data = data


    
    system_prompt = f"""
    You are a smart technical question generation engine.

  
    You will get this as input data 
    
        "practice_type": "<one of ['Backend', 'Frontend', 'DSA', 'DBMS', 'OS', 'NETWORKING']>",
        "difficulty": "<one of ['Easy', 'Medium', 'Hard']>",
        "topic": "<relevant technical topic/tag related to the practice_type>",
        "number_of_questions": "an integer between 1 and 20",
        "question_type": "<one of ['mcq', 'text', 'coding']>"


    Your job is to generate a list of relevant technical questions according to the inputs above.
    This is hghly important test generation make sure the level of difficulty and topic matches the practice type.
    You should generate questions that are relevant to the specified `topic` and match the `difficulty` level.
    ### Rules:
    

    1. **MCQ**: Max 20 questions allowed.
    2. **Coding**: Only valid for these practice types: `'Backend'`, `'Frontend'`, `'DBMS'`, with a strict limit of 3 questions.
    3. **Text**: Short descriptive or conceptual questions allowed for any practice type.
    4. All questions should be relevant to the specified `topic` and match the `difficulty` level.
    5. Coding difficulty should match Leetcode and Codeforces standards.
    7. Coding question should be detailed.
    8. For coding questions, provide a clear problem statement (These problem statements need to be scenario based question which contains real-world context) and problem statements need to be atleast 150 word for difficult, 100 word for medium, 50 word for easy, input/output format, and constraints.

    ### Output Format (JSON):
    if question_type is "mcq" or "coding" or "text" then the output should be in this format:
    only for coding this would be the output format:
    {{
            "type": "coding",
            "questions": [ 
            {{
                "problem statement": "string",
                "input_format": "string",
                "output_format": "string",
                "constraints": "string",
                "Test Cases": [
                    {{
                        "input": "string",
                        "output": "string"
                    }}
                ]
                }}
                
            ],
            "expected_time_to_complete_test": "HH:MM:SS",
           
            }}

    for  MCQ Type:
     "mcq_data": {{
       
        "questions":[
            {{
                "question": "string",
                "Options": [ string, string, string, string ]
                "answer": "string"  # This should be correction option number correct 
                "explaination": "string"  # Explanation for the answer

            }}
        ],
        "expected_time_to_complete_test": "HH:MM:SS",
       
        ]
    }}
  
    
    Strict Guidelines:
                    1. No text outside the JSON object.

                    2. No markdown or code blocks in the JSON object.

                    3. Must be valid JSON open and close braces.

                    4. Structure must be exactly as specified, with no extra fields or changes.

                    5. Do not add ''' or '''json to the JSON object before or after the JSON object.

                    7. You reponse should start with open brace and end with close brace.

                    8. Json object should contain only the fields specified in the output format.
    """

    
    messages = [
    ("system", system_prompt),
    ("human", json.dumps(user_data)), 
  ]

    ai_msg = llm.invoke(messages)
    response = json.loads(ai_msg.content)
  

    return response



def questionAnalyser(data:dict) -> json:
    """
    Analyzes the given resume using the Groq LLM.

    Args:
        text (str): The text to summarize.
    
    Returns:
        json: The analysis results.

    """

    user_data = data


    system_prompt = f"""
    

   
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
    response = json.loads(ai_msg.content)
    

    return response




