import time
import fitz
import tempfile
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from ai.summary import summarize_text, resume_analysis
from ai.QuestionBuilder.questionBuilder import questionBuilderv1, questionAnalyser
from fastapi.middleware.cors import CORSMiddleware
from Models.modelsv1 import QuestionData, QuestionAnalyse
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

import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager

def scrape_with_selenium(url: str) -> str:
    options = uc.ChromeOptions()
    options.headless = True
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0")

    driver = uc.Chrome(
        options=options,
        driver_executable_path=ChromeDriverManager().install()
    )

    try:
        driver.get(url)
        time.sleep(5)  # Replace with WebDriverWait if needed
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    finally:
        driver.quit()



@app.get("/")
def root():
    return {"message": "Welcome to the AI ASTRA. Use the endpoints to interact with the AI services."}

@app.post("/scrape")
def scrape_links(input_data: LinksInput):
    results = []
    for link in input_data.links:
        try:
            text = scrape_with_selenium(link)
            results.append({"source": link, "content": text})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error scraping {link}: {str(e)}")
    return {"ai": summarize_text(results)}

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
   

