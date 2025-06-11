from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from ai.summary import summarize_text, resume_analysis
from bs4 import BeautifulSoup
import time
import fitz
import tempfile
import os

app = FastAPI()

class LinksInput(BaseModel):
    links: List[str]

def scrape_with_selenium(url: str) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    finally:
        driver.quit()

@app.post("/scrape/")
def scrape_links(input_data: LinksInput):
    results = []
    for link in input_data.links:
        try:
            text = scrape_with_selenium(link)
            results.append({"source": link, "content": text})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error scraping {link}: {str(e)}")
    return {"ai": summarize_text(results)}

@app.post("/extract-text/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text("text") for page in doc)
        result = resume_analysis({"text": text})
        doc.close()
        return JSONResponse(content={"filename": file.filename, "resume_analysis": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
