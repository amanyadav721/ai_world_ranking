import time
import fitz
import tempfile
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from ai.summary import summarize_text, resume_analysis
from ai.QuestionBuilder.questionBuilder import questionBuilderv1, questionAnalyser
from fastapi.middleware.cors import CORSMiddleware
from Models.modelsv1 import QuestionData
from bs4 import BeautifulSoup


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specify allowed origins
    allow_methods=["POST", "OPTIONS"],  # Allow POST and OPTIONS
    allow_headers=["Content-Type"],
)

class LinksInput(BaseModel):
    links: List[str]






@app.get("/")
async def root():
    return {"message": "Welcome to the AI ASTRA. Use the endpoints to interact with the AI services."}


@app.post("/ai/resumeranker")
async def extract_text_from_pdf(file: UploadFile = File(...), job_title: str = Form(...), job_description: str = Form(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text("text") for page in doc)
        result = result = resume_analysis(text, job_title, job_description)
        doc.close()
        return JSONResponse(content={"filename": file.filename, "resume_analysis": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    



@app.post("/ai/questionBuilder")
def scrape_links(qd:QuestionData):
    
        try:
            question_data = {
                "practice_type": qd.practice_type,
                "difficulty": qd.difficulty,
                "topic": qd.topic,
                "question_type": qd.question_type,
                "number_of_questions": qd.number_of_questions
            }
            return questionBuilderv1(question_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question data: {str(e)}")
   

