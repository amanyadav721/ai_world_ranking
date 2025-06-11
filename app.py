from fastapi import FastAPI, HTTPException,UploadFile,File
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

port = int(os.environ.get("PORT", 8000))

class LinksInput(BaseModel):
    links: List[str]

def scrape_with_selenium(url: str) -> str:
    # Set up headless Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/58.0.3029.110 Safari/537.3")

    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the URL
        driver.get(url)
        time.sleep(5)  # Wait for the page to load completely

        # Get the page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract the desired information
        text = soup.get_text(separator=' ', strip=True)
        return text
    finally:
        driver.quit()

@app.post("/scrape/")
def scrape_links(input_data: LinksInput):
    results = []
    try:
        try:
            for link in input_data.links:
                try:
                    text = scrape_with_selenium(link)
                    results.append({
                        "source": link,
                        "content": text
                    })
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error scraping {link}: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

        response = {"ai": summarize_text(results)}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/extract-text/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract text from the PDF
        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text("text") for page in doc)
        resume_analysis_result = resume_analysis({"text": text})

        doc.close()

        return JSONResponse(content={"filename": file.filename, "resume_analysis":resume_analysis_result})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)