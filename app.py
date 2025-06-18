import shutil
import time
import fitz
import tempfile
import os
from git import Repo
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from ai.summary import summarize_text, resume_analysis
from ai.QuestionBuilder.questionBuilder import questionBuilderv1, questionAnalyser
from ai.LLDai.lld_ai import lld_creator
from fastapi.middleware.cors import CORSMiddleware
from Models.modelsv1 import QuestionData
from bs4 import BeautifulSoup
from ai.githubai.github_handler import clone_repo, extract_code_from_repo
from ai.githubai.doc_generator import generate_doc_from_code



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify frontend origin like "https://yourfrontend.com"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        print("job_title:", job_title, "job_description:", job_description)

        try:
            doc = fitz.open(tmp_path)
            text = "\n".join(page.get_text("text") for page in doc)
            print(text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading PDF file: {str(e)}")
        
        result = resume_analysis(text, job_title, job_description)
        print("Result:", result)
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
        

@app.post("/ai/lld")
def lld_ai(qd:dict):
    
        try:
            return lld_creator(qd)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question data: {str(e)}")
   















import tempfile
from contextlib import contextmanager



@contextmanager
def clone_repo_to_temp(repo_url: str):
    """
    A context manager to clone a repo into a temporary directory
    and ensure it's cleaned up automatically.
    """
    temp_dir = tempfile.mkdtemp(prefix="repo_")
    try:
        print(f"Cloning {repo_url} into temporary directory: {temp_dir}")
        Repo.clone_from(repo_url, temp_dir)
        yield temp_dir
    except Exception as e:
        # Re-raise exceptions to be caught by the endpoint handler
        raise ConnectionError(f"Failed to clone repository: {e}")
    finally:
        # This 'finally' block ensures cleanup happens even if errors occur
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        

@app.post("/generate-doc/")
def generate_documentation(repo_url: str = Query(..., description="Public GitHub repository URL")):
    """
    Main endpoint to clone a repo, extract code, and generate documentation.
    """
    try:
        with clone_repo_to_temp(repo_url) as repo_path:
            code = extract_code_from_repo(repo_path)
            if not code:
                raise HTTPException(status_code=404, detail="No supported code files found in the repository.")
            
            documentation = generate_doc_from_code(code)
            return {"documentation": documentation}
            
    except ConnectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
