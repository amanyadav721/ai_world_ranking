from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from ai.summary import summarize_text
from bs4 import BeautifulSoup
import time

app = FastAPI()

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
