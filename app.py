from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

class LinksInput(BaseModel):
    links: List[str]

def scrape_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the page
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping {url}: {str(e)}")

@app.post("/scrape/")
def scrape_links(input_data: LinksInput):
    documents = []
    for link in input_data.links:
        text = scrape_text_from_url(link)
        documents.append(Document(page_content=text, metadata={"source": link}))
    
    # Use LangChain's text splitter to process documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    
    # Prepare the response
    result = [
        {
            "source": doc.metadata["source"],
            "content": doc.page_content
        }
        for doc in split_docs
    ]
    return result
