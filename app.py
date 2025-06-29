import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
import uuid
from typing import List

import fitz
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from git import Repo
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
# from ai.centralAI.agents import smartAgentOnboardingStreaming
# from fastapi.responses import StreamingResponse
from pdfminer.high_level import extract_text
from pydantic import BaseModel

from ai.chatwithPDF.chat_pdf import (embedding_model, get_chat_model,
                                     get_hashed_email_path,
                                     load_or_create_vector_store,
                                     user_data_cache)
from ai.githubai.doc_generator import generate_doc_from_code
from ai.githubai.github_handler import clone_repo, extract_code_from_repo
from ai.LLDai.lld_ai import lld_creator
from ai.QuestionBuilder.questionBuilder import (questionAnalyser,
                                                questionBuilderv1)
from ai.summary import resume_analysis
from Models.modelsv1 import (ChatRequest, ChatResponse, QuestionData,
                             UploadResponse, FinancialAnalysisRequest, FinancialAnalysisResponse)
from ai.financial_analysis import analyze_financial_data

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
    return {
        "message": "Welcome to the AI ASTRA. Use the endpoints to interact with the AI services."
    }


# class QuestionRequest(BaseModel):
#     user: str
#     email: str

# @app.post("/stream/ai/2")
# async def ai_endpoint(data: QuestionRequest):
#     return StreamingResponse(
#         smartAgentOnboardingStreaming(data.user, data.email),
#         media_type="application/json",
#     )


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII chars
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()


from typing import Any


def sanitize_and_parse_json(raw: str) -> Any:
    """
    Cleans and parses a possibly malformed JSON string from an LLM.
    """
    try:
        # Remove code block markers and invalid control characters
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"[\x00-\x1F\x7F]", "", raw)  # control chars
        raw = raw.replace("\n", "\\n").replace("\r", "")  # escape newlines

        return json.loads(raw)
    except json.JSONDecodeError as e:
        print("❌ JSON Decode Failed. Raw content:\n", raw[:1000])
        raise ValueError(f"resume_analysis failed: {e}")


@app.post("/ai/resumeranker")
async def extract_text_from_pdf(
    file: UploadFile = File(...),
    job_title: str = Form(...),
    job_description: str = Form(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            text = extract_text(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error extracting text from PDF: {str(e)}"
            )

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="PDF contains no extractable text."
            )

        clean_resume_text = clean_text(text)

        try:
            analysis_result = resume_analysis(
                clean_resume_text, job_title, job_description
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500, detail=f"AI model JSON parsing failed: {str(e)}"
            )

        return {"filename": file.filename, "resume_analysis": analysis_result}

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ai/questionBuilder")
def scrape_links(qd: QuestionData):

    try:
        question_data = {
            "practice_type": qd.practice_type,
            "difficulty": qd.difficulty,
            "topic": qd.topic,
            "question_type": qd.question_type,
            "number_of_questions": qd.number_of_questions,
        }
        return questionBuilderv1(question_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing question data: {str(e)}"
        )


@app.post("/ai/lld")
def lld_ai(qd: dict):

    try:
        return lld_creator(qd)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing question data: {str(e)}"
        )


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
def generate_documentation(
    repo_url: str = Query(..., description="Public GitHub repository URL")
):
    """
    Main endpoint to clone a repo, extract code, and generate documentation.
    """
    try:
        with clone_repo_to_temp(repo_url) as repo_path:
            code = extract_code_from_repo(repo_path)
            if not code:
                raise HTTPException(
                    status_code=404,
                    detail="No supported code files found in the repository.",
                )

            documentation = generate_doc_from_code(code)
            return {"documentation": documentation}

    except ConnectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


# -----------------------Chat with PDF----------------------------#
@app.post("/upload-pdf/", response_model=UploadResponse)
async def upload_pdf(
    user_email: str = Form(..., description="The email of the user."),
    file: UploadFile = File(..., description="The PDF file to process."),
):
    """
    Handles PDF upload, processing, and vector store creation for a specific user.
    If a PDF was previously uploaded for this email, it will be replaced.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF."
        )

    user_email_hash_path = get_hashed_email_path(user_email)
    temp_pdf_path = f"temp_{os.path.basename(user_email_hash_path)}.pdf"

    try:
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vector_store = load_or_create_vector_store(temp_pdf_path, user_email_hash_path)

        # Initialize or reset the ConversationBufferMemory for this user
        user_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        user_data_cache[user_email] = {
            "vector_store": vector_store,
            "memory": user_memory,  # Store the memory object
            "faiss_path": user_email_hash_path,
        }

        return {
            "status": "success",
            "message": f"PDF '{file.filename}' uploaded and processed for {user_email}.",
            "user_email": user_email,
        }
    except Exception as e:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        await file.close()


@app.post("/chat/", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """
    Handles the chat conversation for a specific user.
    """
    user_email = request.user_email
    user_email_hash_path = get_hashed_email_path(user_email)

    # Check if the user's data is in cache, if not, try to load from disk
    if user_email not in user_data_cache:
        if os.path.exists(user_email_hash_path) and os.path.isdir(user_email_hash_path):
            try:
                print(
                    f"Loading vector store from disk for {user_email} (not in cache)..."
                )
                vector_store = FAISS.load_local(
                    user_email_hash_path,
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
                # When loading from disk, we also need to re-initialize the memory if it's not persisted
                # In this example, memory is NOT persisted, so it starts fresh.
                user_memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )

                user_data_cache[user_email] = {
                    "vector_store": vector_store,
                    "memory": user_memory,  # Initialize new memory object
                }
                print("Vector store loaded into cache.")
            except Exception as e:
                print(f"Failed to load vector store from disk for {user_email}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail="No processed PDF found or could not load for this user. Please upload a PDF first.",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No PDF processed for this user. Please upload a PDF first via the /upload-pdf/ endpoint.",
            )

    session_data = user_data_cache[user_email]
    current_vector_store = session_data["vector_store"]
    current_memory = session_data["memory"]  # Get the stored memory object

    try:
        # Create a conversational chain, passing the SAME memory object
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=get_chat_model(),
            retriever=current_vector_store.as_retriever(),
            memory=current_memory,  # <<< THIS IS THE CRUCIAL CHANGE
        )

        # Get the answer from the chain
        # You no longer pass chat_history directly to the chain's __call__
        # The memory object inside the chain handles it automatically.
        result = conversation_chain({"question": request.query})
        answer = result["answer"]

        # The memory object is automatically updated by the chain; no manual append needed here.

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    """Clean up all temporary files on server shutdown."""
    print("Server shutting down. Cleaning up temporary files.")
    # In this version, FAISS indices are persisted in FAISS_STORAGE_DIR.
    # We only clean up the temporary PDF files if any remain.
    # For a full cleanup (e.g., development reset), you might remove FAISS_STORAGE_DIR manually.
    for filename in os.listdir("."):
        if filename.startswith("temp_") and filename.endswith(".pdf"):
            os.remove(filename)
    print("Temporary PDF files cleaned up.")


# -----------------------Free Services----------------------------#
import shutil
import tempfile
import zipfile
from io import BytesIO
from typing import List, Optional

import PyPDF2
import pypdfium2 as pdfium
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, UploadFile,
                     status)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont


def get_temp_paths(suffix: str) -> tuple[str, str]:
    """Generates a temporary file path and its containing directory path."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, f"output{suffix}")
    return temp_file, temp_dir


def cleanup_temp_dir(temp_dir_path: str):
    """Removes the temporary directory."""
    if os.path.exists(temp_dir_path):
        shutil.rmtree(temp_dir_path)
        print(f"Cleaned up temporary directory: {temp_dir_path}")


# --- API Endpoints ---


@app.post("/pdf/minimize", summary="Minimize PDF Size")
async def minimize_pdf(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):  # Add background_tasks as a dependency
    """
    Minimizes the size of an uploaded PDF file.
    Note: PyPDF2's minimization is basic (re-saving with some compression).
    More aggressive compression might require external tools like Ghostscript.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed for minimization.",
        )

    # Create temporary input and output files and their parent directories
    temp_input_path, temp_input_dir = get_temp_paths(suffix=".pdf")
    temp_output_path, temp_output_dir = get_temp_paths(suffix="_minimized.pdf")

    try:
        # Save the uploaded file to a temporary location
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        reader = PyPDF2.PdfReader(temp_input_path)
        writer = PyPDF2.PdfWriter()

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            # PyPDF2's compress_content_streams attempts to compress content
            # streams within the page, which can reduce file size.
            page.compress_content_streams()
            writer.add_page(page)

        with open(temp_output_path, "wb") as output_pdf:
            writer.write(output_pdf)

        # Add cleanup tasks to run in the background after the response is sent
        background_tasks.add_task(cleanup_temp_dir, temp_output_dir)
        background_tasks.add_task(cleanup_temp_dir, temp_input_dir)

        return FileResponse(
            path=temp_output_path,
            filename=f"minimized_{file.filename}",
            media_type="application/pdf",
        )
    except Exception as e:
        # Clean up input and output temp dirs if an error occurs
        cleanup_temp_dir(temp_input_dir)
        cleanup_temp_dir(temp_output_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to minimize PDF: {e}",
        )


@app.post("/image/optimize", summary="Optimize Image Size and Format")
async def optimize_image(
    background_tasks: BackgroundTasks,  # Add background_tasks
    file: UploadFile = File(...),
    quality: int = 80,  # For JPEG/WEBP, 0-100 scale
    output_format: Optional[str] = None,  # e.g., "JPEG", "PNG", "WEBP"
):
    """
    Optimizes an uploaded image by adjusting quality and/or changing its format.
    Accepts various image types (JPEG, PNG, GIF, BMP, TIFF, WebP, etc.).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are allowed for optimization.",
        )

    # Determine output format and suffix
    if output_format:
        output_format = output_format.upper()
        if output_format not in ["JPEG", "PNG", "WEBP", "GIF", "TIFF", "BMP"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid output format. Choose from JPEG, PNG, WEBP, GIF, TIFF, BMP.",
            )
        suffix = f".{output_format.lower()}"
    else:
        # Use original format if not specified
        suffix = os.path.splitext(file.filename)[1]
        if not suffix:  # Fallback if no extension
            suffix = ".jpg"  # Default to jpg if no extension found
        output_format = file.content_type.split("/")[
            -1
        ].upper()  # Try to infer from content type

    temp_output_path, temp_output_dir = get_temp_paths(suffix=suffix)

    try:
        image = Image.open(BytesIO(await file.read()))

        # Convert to RGB if saving as JPEG (Pillow requirement)
        if output_format == "JPEG" and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        elif (
            output_format == "PNG" and image.mode == "P"
        ):  # Ensure PNG handles transparency properly
            image = image.convert("RGBA")

        # Save with optimization settings
        save_params = {"quality": quality} if output_format in ["JPEG", "WEBP"] else {}
        if output_format == "PNG":
            save_params["optimize"] = True  # PNG specific optimization

        image.save(temp_output_path, format=output_format, **save_params)

        background_tasks.add_task(cleanup_temp_dir, temp_output_dir)

        return FileResponse(
            path=temp_output_path,
            filename=f"optimized_{os.path.splitext(file.filename)[0]}{suffix}",
            media_type=f"image/{output_format.lower()}",
        )
    except Exception as e:
        cleanup_temp_dir(temp_output_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize image: {e}",
        )


@app.post("/image/to-pdf", summary="Convert Images to PDF")
async def images_to_pdf(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):  # Add background_tasks
    """
    Converts one or more uploaded image files into a single PDF document.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files provided.",
        )

    images = []
    # No direct temp files needed for input images as they are read into BytesIO

    for uploaded_file in files:
        if not uploaded_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{uploaded_file.filename}' is not an image. Only image files are allowed.",
            )
        try:
            image = Image.open(BytesIO(await uploaded_file.read()))
            # Ensure images are in RGB mode for PDF saving (no alpha channel in typical PDFs)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            images.append(image)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not process image '{uploaded_file.filename}': {e}",
            )

    if not images:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid images found to convert to PDF.",
        )

    temp_output_path, temp_output_dir = get_temp_paths(suffix=".pdf")

    try:
        # Save the first image, appending others if multiple
        images[0].save(
            temp_output_path,
            save_all=True,
            append_images=images[1:] if len(images) > 1 else None,
            resolution=100.0,  # DPI for the PDF
            quality=95,  # Quality for images inside the PDF
        )

        background_tasks.add_task(cleanup_temp_dir, temp_output_dir)

        return FileResponse(
            path=temp_output_path,
            filename="converted_images.pdf",
            media_type="application/pdf",
        )
    except Exception as e:
        cleanup_temp_dir(temp_output_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert images to PDF: {e}",
        )


@app.post("/pdf/to-images", summary="Convert PDF to Images (per page)")
async def pdf_to_images(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):  # Add background_tasks
    """
    Converts each page of an uploaded PDF file into separate image files (PNG)
    and returns them in a ZIP archive.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed for conversion to images.",
        )

    temp_input_path, temp_input_dir = get_temp_paths(suffix=".pdf")
    temp_zip_path, temp_zip_dir = get_temp_paths(suffix=".zip")
    temp_images_dir = tempfile.mkdtemp()  # Directory to store individual images

    try:
        # Save the uploaded file to a temporary location
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc = pdfium.PdfDocument(temp_input_path)

        image_paths = []
        for i in range(len(doc)):
            page = doc[i]
            # Render page to a Pillow image
            pil_image = page.render_to_image(
                scale=2  # Render at 2x resolution for better quality
            )
            image_filename = f"page_{i+1}.png"
            image_path = os.path.join(temp_images_dir, image_filename)
            pil_image.save(image_path, format="PNG")
            image_paths.append(image_path)

        # Create a zip file containing all generated images
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_path in image_paths:
                # Add file to zip, preserving only the filename
                zf.write(img_path, os.path.basename(img_path))

        # Add cleanup tasks
        background_tasks.add_task(cleanup_temp_dir, temp_zip_dir)
        background_tasks.add_task(cleanup_temp_dir, temp_input_dir)
        background_tasks.add_task(
            cleanup_temp_dir, temp_images_dir
        )  # Clean up the images directory

        return FileResponse(
            path=temp_zip_path,
            filename=f"{os.path.splitext(file.filename)[0]}_images.zip",
            media_type="application/zip",
        )
    except Exception as e:
        print(f"Error converting PDF to images: {e}")  # Debugging
        cleanup_temp_dir(temp_input_dir)
        cleanup_temp_dir(temp_zip_dir)
        if os.path.exists(temp_images_dir):  # Ensure cleanup even if zip not created
            shutil.rmtree(temp_images_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert PDF to images: {e}",
        )


@app.post("/image/watermark", summary="Add Text Watermark to Image")
async def watermark_image(
    background_tasks: BackgroundTasks,  # Add background_tasks
    file: UploadFile = File(...),
    watermark_text: str = "CONFIDENTIAL",
    opacity: int = 128,  # 0-255, where 255 is fully opaque
    font_size: int = 50,
):
    """
    Adds a text watermark to an uploaded image.
    The watermark will be centered on the image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are allowed for watermarking.",
        )
    if not (0 <= opacity <= 255):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Opacity must be between 0 and 255.",
        )
    if font_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Font size must be greater than 0.",
        )

    temp_output_path, temp_output_dir = get_temp_paths(
        suffix=os.path.splitext(file.filename)[1]
    )

    try:
        # Open the image
        img = Image.open(BytesIO(await file.read())).convert(
            "RGBA"
        )  # Convert to RGBA for alpha channel handling

        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # Try to load a default font that is usually available
        try:
            # Arial or DejaVuSans is common on Linux, Arial.ttf on Windows
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype(
                    "DejaVuSans-Bold.ttf", font_size
                )  # Common on many Linux systems
            except IOError:
                font = (
                    ImageFont.load_default()
                )  # Fallback to default if no TrueType font found
                print(
                    "Warning: Could not find Arial.ttf or DejaVuSans-Bold.ttf. Using default font."
                )

        # Calculate text size and position
        # Use getbbox for more accurate text size
        text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = (img.width - text_width) / 2
        y = (img.height - text_height) / 2

        # Create a transparent overlay for the watermark
        watermark_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw_watermark = ImageDraw.Draw(watermark_layer)

        # Draw the text with desired color (e.g., light grey) and opacity
        text_color = (192, 192, 192, opacity)  # RGBA: light grey with given opacity
        draw_watermark.text((x, y), watermark_text, font=font, fill=text_color)

        # Composite the watermark layer onto the original image
        watermarked_img = Image.alpha_composite(img, watermark_layer)

        # Save the watermarked image, using the original format if possible
        # Or convert to RGB if the output format doesn't support transparency (e.g., JPEG)
        original_format = file.content_type.split("/")[-1].upper()
        if original_format == "JPEG":
            watermarked_img = watermarked_img.convert("RGB")
            save_params = {"quality": 90}
        else:
            save_params = {}

        watermarked_img.save(temp_output_path, format=original_format, **save_params)

        background_tasks.add_task(cleanup_temp_dir, temp_output_dir)

        return FileResponse(
            path=temp_output_path,
            filename=f"watermarked_{file.filename}",
            media_type=file.content_type,
        )
    except Exception as e:
        cleanup_temp_dir(temp_output_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to watermark image: {e}",
        )


@app.get("/test", response_class=HTMLResponse)
async def test_endpoint():
    """
    A simple test endpoint to verify the API is running.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FastAPI File Processing Tools Test</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
            color: #333;
        }
        .container {
            max-width: 960px;
            margin: 2rem auto;
            padding: 1.5rem;
            background-color: #ffffff;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .service-card {
            background-color: #f9fafb;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e5e7eb;
        }
        .service-card h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: #4b5563;
        }
        .form-group input[type="file"],
        .form-group input[type="text"],
        .form-group input[type="number"],
        .form-group select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 0.5rem;
            box-sizing: border-box; /* Ensures padding doesn't increase width */
            background-color: #ffffff;
            transition: border-color 0.2s;
        }
        .form-group input[type="file"]:focus,
        .form-group input[type="text"]:focus,
        .form-group input[type="number"]:focus,
        .form-group select:focus {
            outline: none;
            border-color: #4f46e5; /* Indigo-600 */
        }
        .btn-submit {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background-color: #4f46e5; /* Indigo-600 */
            color: #ffffff;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s ease-in-out;
        }
        .btn-submit:hover {
            background-color: #4338ca; /* Indigo-700 */
        }
        .message {
            margin-top: 1rem;
            padding: 0.75rem;
            border-radius: 0.5rem;
            font-weight: 500;
        }
        .message.success {
            background-color: #d1fae5; /* Green-100 */
            color: #065f46; /* Green-700 */
            border: 1px solid #a7f3d0; /* Green-200 */
        }
        .message.error {
            background-color: #fee2e2; /* Red-100 */
            color: #b91c1c; /* Red-700 */
            border: 1px solid #fca5a5; /* Red-200 */
        }
        .loading-spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-top: 4px solid #4f46e5;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-left: 0.5rem;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .result-area {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }
        .result-area a {
            color: #4f46e5;
            text-decoration: underline;
            font-weight: 600;
        }
        .result-image {
            max-width: 100%;
            height: auto;
            border-radius: 0.5rem;
            margin-top: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">FastAPI File Processing Tools Tester</h1>

        <!-- PDF Minimize Service -->
        <div class="service-card">
            <h2>PDF Minimizer</h2>
            <form id="pdfMinimizeForm">
                <div class="form-group">
                    <label for="pdfMinimizeFile">Upload PDF:</label>
                    <input type="file" id="pdfMinimizeFile" name="file" accept="application/pdf" required class="cursor-pointer">
                </div>
                <button type="submit" class="btn-submit">
                    Minimize PDF
                    <span id="pdfMinimizeLoading" class="loading-spinner hidden"></span>
                </button>
                <div id="pdfMinimizeMessage" class="message hidden"></div>
                <div id="pdfMinimizeResult" class="result-area"></div>
            </form>
        </div>

        <!-- Image Optimizer Service -->
        <div class="service-card">
            <h2>Image Optimizer</h2>
            <form id="imageOptimizeForm">
                <div class="form-group">
                    <label for="imageOptimizeFile">Upload Image:</label>
                    <input type="file" id="imageOptimizeFile" name="file" accept="image/*" required class="cursor-pointer">
                </div>
                <div class="form-group">
                    <label for="imageOptimizeQuality">Quality (0-100, for JPEG/WEBP):</label>
                    <input type="number" id="imageOptimizeQuality" name="quality" value="80" min="0" max="100">
                </div>
                <div class="form-group">
                    <label for="imageOptimizeFormat">Output Format (Optional):</label>
                    <select id="imageOptimizeFormat" name="output_format">
                        <option value="">Original</option>
                        <option value="JPEG">JPEG</option>
                        <option value="PNG">PNG</option>
                        <option value="WEBP">WEBP</option>
                        <option value="GIF">GIF</option>
                        <option value="TIFF">TIFF</option>
                        <option value="BMP">BMP</option>
                    </select>
                </div>
                <button type="submit" class="btn-submit">
                    Optimize Image
                    <span id="imageOptimizeLoading" class="loading-spinner hidden"></span>
                </button>
                <div id="imageOptimizeMessage" class="message hidden"></div>
                <div id="imageOptimizeResult" class="result-area"></div>
            </form>
        </div>

        <!-- Image to PDF Converter Service -->
        <div class="service-card">
            <h2>Image to PDF Converter</h2>
            <form id="imageToPdfForm">
                <div class="form-group">
                    <label for="imageToPdfFiles">Upload Images (select multiple):</label>
                    <input type="file" id="imageToPdfFiles" name="files" accept="image/*" multiple required class="cursor-pointer">
                </div>
                <button type="submit" class="btn-submit">
                    Convert to PDF
                    <span id="imageToPdfLoading" class="loading-spinner hidden"></span>
                </button>
                <div id="imageToPdfMessage" class="message hidden"></div>
                <div id="imageToPdfResult" class="result-area"></div>
            </form>
        </div>

        <!-- PDF to Images Converter Service -->
        <div class="service-card">
            <h2>PDF to Images Converter</h2>
            <form id="pdfToImagesForm">
                <div class="form-group">
                    <label for="pdfToImagesFile">Upload PDF:</label>
                    <input type="file" id="pdfToImagesFile" name="file" accept="application/pdf" required class="cursor-pointer">
                </div>
                <button type="submit" class="btn-submit">
                    Convert to Images (ZIP)
                    <span id="pdfToImagesLoading" class="loading-spinner hidden"></span>
                </button>
                <div id="pdfToImagesMessage" class="message hidden"></div>
                <div id="pdfToImagesResult" class="result-area"></div>
            </form>
        </div>

        <!-- Image Watermark Service -->
        <div class="service-card">
            <h2>Image Watermarker</h2>
            <form id="imageWatermarkForm">
                <div class="form-group">
                    <label for="imageWatermarkFile">Upload Image:</label>
                    <input type="file" id="imageWatermarkFile" name="file" accept="image/*" required class="cursor-pointer">
                </div>
                <div class="form-group">
                    <label for="watermarkText">Watermark Text:</label>
                    <input type="text" id="watermarkText" name="watermark_text" value="CONFIDENTIAL" placeholder="Enter watermark text">
                </div>
                <div class="form-group">
                    <label for="watermarkOpacity">Opacity (0-255):</label>
                    <input type="number" id="watermarkOpacity" name="opacity" value="128" min="0" max="255">
                </div>
                <div class="form-group">
                    <label for="watermarkFontSize">Font Size:</label>
                    <input type="number" id="watermarkFontSize" name="font_size" value="50" min="1">
                </div>
                <button type="submit" class="btn-submit">
                    Add Watermark
                    <span id="imageWatermarkLoading" class="loading-spinner hidden"></span>
                </button>
                <div id="imageWatermarkMessage" class="message hidden"></div>
                <div id="imageWatermarkResult" class="result-area"></div>
            </form>
        </div>
    </div>

    <script>
        const API_BASE_URL = 'http://127.0.0.1:8000'; // Make sure this matches your FastAPI server address

        /**
         * Helper function to show messages (success/error)
         * @param {HTMLElement} messageElement - The div element to display the message in.
         * @param {string} text - The message text.
         * @param {string} type - 'success' or 'error'.
         */
        function showMessage(messageElement, text, type) {
            messageElement.textContent = text;
            messageElement.className = `message ${type}`;
            messageElement.classList.remove('hidden');
        }

        /**
         * Helper function to hide messages
         * @param {HTMLElement} messageElement - The div element whose message needs to be hidden.
         */
        function hideMessage(messageElement) {
            messageElement.classList.add('hidden');
            messageElement.textContent = '';
        }

        /**
         * Helper function to toggle loading spinner
         * @param {HTMLElement} spinnerElement - The spinner element.
         * @param {boolean} show - True to show, false to hide.
         */
        function toggleLoading(spinnerElement, show) {
            if (show) {
                spinnerElement.classList.remove('hidden');
            } else {
                spinnerElement.classList.add('hidden');
            }
        }

        /**
         * Generic function to handle form submissions
         * @param {HTMLFormElement} form - The form element.
         * @param {string} endpoint - The FastAPI endpoint URL.
         * @param {string} method - HTTP method ('POST').
         * @param {HTMLElement} messageElement - Element to show success/error messages.
         * @param {HTMLElement} resultElement - Element to display results (e.g., download link).
         * @param {HTMLElement} loadingSpinner - Element for the loading spinner.
         * @param {boolean} isMultiFile - True if the input is multiple files.
         * @param {string} expectedMediaTypePrefix - e.g., 'application/pdf', 'image/', 'application/zip'.
         */
        async function handleSubmit(form, endpoint, messageElement, resultElement, loadingSpinner, isMultiFile = false, expectedMediaTypePrefix = '') {
            hideMessage(messageElement);
            resultElement.innerHTML = ''; // Clear previous results
            toggleLoading(loadingSpinner, true);

            const formData = new FormData();
            let fileInput;

            if (isMultiFile) {
                fileInput = form.querySelector('input[type="file"][multiple]');
                if (fileInput && fileInput.files.length > 0) {
                    for (const file of fileInput.files) {
                        formData.append(fileInput.name, file);
                    }
                } else {
                    showMessage(messageElement, 'Please select at least one file.', 'error');
                    toggleLoading(loadingSpinner, false);
                    return;
                }
            } else {
                fileInput = form.querySelector('input[type="file"]:not([multiple])');
                if (fileInput && fileInput.files.length > 0) {
                    formData.append(fileInput.name, fileInput.files[0]);
                } else {
                    showMessage(messageElement, 'Please select a file.', 'error');
                    toggleLoading(loadingSpinner, false);
                    return;
                }
            }

            // Append other form fields (text, number, select)
            const inputs = form.querySelectorAll('input:not([type="file"]), select');
            inputs.forEach(input => {
                if (input.name && input.value) { // Only append if name and value exist
                    formData.append(input.name, input.value);
                }
            });

            try {
                const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const filename = response.headers.get('content-disposition')?.split('filename=')[1]?.replace(/"/g, '') || `download.${expectedMediaTypePrefix.split('/')[1] || 'bin'}`;
                    const url = URL.createObjectURL(blob);

                    if (expectedMediaTypePrefix.startsWith('image/') && blob.type.startsWith('image/')) {
                        // Display image directly if it's an image
                        const img = document.createElement('img');
                        img.src = url;
                        img.alt = filename;
                        img.className = 'result-image';
                        resultElement.appendChild(img);
                        showMessage(messageElement, 'Image processed successfully!', 'success');
                    } else {
                        // For other file types, provide a download link
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = filename;
                        link.textContent = `Download Processed File: ${filename}`;
                        link.className = 'text-blue-600 hover:text-blue-800 font-semibold';
                        resultElement.appendChild(link);
                        showMessage(messageElement, 'File processed successfully!', 'success');
                    }
                } else {
                    const errorData = await response.json();
                    showMessage(messageElement, `Error: ${errorData.detail || 'Unknown error occurred.'}`, 'error');
                }
            } catch (error) {
                console.error('Fetch error:', error);
                showMessage(messageElement, `Network error or API is unreachable: ${error.message}`, 'error');
            } finally {
                toggleLoading(loadingSpinner, false);
            }
        }

        // --- Attach Event Listeners ---

        document.getElementById('pdfMinimizeForm').addEventListener('submit', function(e) {
            e.preventDefault();
            handleSubmit(
                this,
                '/pdf/minimize',
                document.getElementById('pdfMinimizeMessage'),
                document.getElementById('pdfMinimizeResult'),
                document.getElementById('pdfMinimizeLoading'),
                false,
                'application/pdf'
            );
        });

        document.getElementById('imageOptimizeForm').addEventListener('submit', function(e) {
            e.preventDefault();
            handleSubmit(
                this,
                '/image/optimize',
                document.getElementById('imageOptimizeMessage'),
                document.getElementById('imageOptimizeResult'),
                document.getElementById('imageOptimizeLoading'),
                false,
                'image/' // Use prefix as type can vary
            );
        });

        document.getElementById('imageToPdfForm').addEventListener('submit', function(e) {
            e.preventDefault();
            handleSubmit(
                this,
                '/image/to-pdf',
                document.getElementById('imageToPdfMessage'),
                document.getElementById('imageToPdfResult'),
                document.getElementById('imageToPdfLoading'),
                true, // This is a multi-file upload
                'application/pdf'
            );
        });

        document.getElementById('pdfToImagesForm').addEventListener('submit', function(e) {
            e.preventDefault();
            handleSubmit(
                this,
                '/pdf/to-images',
                document.getElementById('pdfToImagesMessage'),
                document.getElementById('pdfToImagesResult'),
                document.getElementById('pdfToImagesLoading'),
                false,
                'application/zip'
            );
        });

        document.getElementById('imageWatermarkForm').addEventListener('submit', function(e) {
            e.preventDefault();
            handleSubmit(
                this,
                '/image/watermark',
                document.getElementById('imageWatermarkMessage'),
                document.getElementById('imageWatermarkResult'),
                document.getElementById('imageWatermarkLoading'),
                false,
                'image/' // Use prefix as type can vary
            );
        });
    </script>
</body>
</html>
"""


# Instructions on how to run this application:
# 1. Save the code above as `main.py`.
# 2. Create a `requirements.txt` file with the following content:
#    fastapi
#    uvicorn
#    python-multipart
#    Pillow
#    PyPDF2
#    pypdfium2
# 3. Install the dependencies: `pip install -r requirements.txt`
# 4. Run the FastAPI application: `uvicorn main:app --reload`
# 5. Access the API documentation at `http://127.0.0.1:8000/docs` in your browser.

@app.post("/ai/financial-analysis", response_model=FinancialAnalysisResponse)
def financial_analysis_endpoint(data: FinancialAnalysisRequest):
    try:
        result = analyze_financial_data(data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial analysis failed: {str(e)}")
