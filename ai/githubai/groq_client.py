import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # adjust if using a different variant
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def ask_groq(prompt: str) -> str:
    """Sends a prompt to the Groq AI and returns the response."""
    messages = [
        SystemMessage(content="You are code documentation expert, you will user's code, you task is to provide clear and structured documentation."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return response.content.strip()
