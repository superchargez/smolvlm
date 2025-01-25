from time import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
import io
import uuid
from datetime import datetime
import fitz  # PyMuPDF
import pypandoc
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model globally during app startup
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
quantization_config = BitsAndBytesConfig(load_in_8bit=True) # Consider keeping quantization for memory efficiency
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    # quantization_config=quantization_config, # Uncomment to use quantization if needed
    # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager", # Consider flash attention if on CUDA and compatible GPU
).to(DEVICE)

app = FastAPI(title="Image to Text with SmolVLM")

# In-memory storage for task status and results
task_results = {}

def resize_image_if_needed(image):
    """
    Resizes an image if it's larger than 512x512 pixels, maintaining aspect ratio
    and aiming for optimal quality for the SmolVLM model.
    """
    width, height = image.size
    if width <= 512 and height <= 512:
        return image
    if width > height:
        new_width = 512
        new_height = int(height * (512 / width))
    else:
        new_height = 512
        new_width = int(width * (512 / height))
    resized_image_np = cv2.resize(np.array(image), (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_image_pil = Image.fromarray(resized_image_np)
    return resized_image_pil

def convert_pdf_to_png_bytes(pdf_content):
    """
    Converts the first page of a PDF content to PNG format in bytes.
    """
    try:
        pdf_document = fitz.open(stream=io.BytesIO(pdf_content), filetype='pdf')
        page = pdf_document[0]  # Get the first page
        pix = page.get_pixmap(dpi=300)  # Specify DPI for better quality

        # Ensure the pixmap is in RGB format
        if pix.n < 3:  # If less than 3 color channels
            pix = fitz.Pixmap(pix, 0)  # Convert to RGB

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_bytes = png_buffer.getvalue()
        return png_bytes, img.size # Return png bytes and image size
    except Exception as e:
        raise Exception(f"Error converting PDF to PNG: {str(e)}")

def convert_docx_to_pdf_bytes(docx_content, filename="document.docx"):
    """
    Converts DOCX content to PDF content in bytes using pypandoc.
    """
    try:
        docx_temp_file = f"/tmp/{filename}"
        with open(docx_temp_file, "wb") as f:
            f.write(docx_content)
        pdf_content = pypandoc.convert_file(docx_temp_file, 'pdf', format='docx')
        os.remove(docx_temp_file) # Clean up temp file
        return pdf_content.encode('utf-8') # Return PDF content as bytes
    except Exception as e:
        raise Exception(f"Error converting DOCX to PDF: {str(e)}")

def convert_pptx_to_pdf_bytes(pptx_content, filename="presentation.pptx"):
    """
    Converts PPTX content to PDF content in bytes using pypandoc.
    """
    try:
        pptx_temp_file = f"/tmp/{filename}"
        with open(pptx_temp_file, "wb") as f:
            f.write(pptx_content)
        pdf_content = pypandoc.convert_file(pptx_temp_file, 'pdf', format='pptx')
        os.remove(pptx_temp_file) # Clean up temp file
        return pdf_content.encode('utf-8') # Return PDF content as bytes
    except Exception as e:
        raise Exception(f"Error converting PPTX to PDF: {str(e)}")


def process_image_description(file_content, task_id, filename, mime_type, size_bytes):
    """
    Processes the file (image or PDF or converted PDF from docx/pptx) to generate a description and stores the result with detailed info.
    This function runs in the background.
    """
    start_process_time = time()
    start_datetime = datetime.utcnow().isoformat() + 'Z' # ISO format with UTC timezone
    processed_format = mime_type # Initial processed format is same as mime_type, will be updated if conversion happens
    original_width, original_height = None, None # Initialize original dimensions

    try:
        if mime_type == "application/pdf":
            processed_format = "PNG (from PDF)"
            png_bytes, original_size = convert_pdf_to_png_bytes(file_content)
            original_width, original_height = original_size
            image = Image.open(io.BytesIO(png_bytes))
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": # DOCX
            processed_format = "PNG (from DOCX)"
            pdf_content = convert_docx_to_pdf_bytes(file_content, filename)
            png_bytes, original_size = convert_pdf_to_png_bytes(pdf_content)
            original_width, original_height = original_size
            image = Image.open(io.BytesIO(png_bytes))
        elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation": # PPTX
            processed_format = "PNG (from PPTX)"
            pdf_content = convert_pptx_to_pdf_bytes(file_content, filename)
            png_bytes, original_size = convert_pdf_to_png_bytes(pdf_content)
            original_width, original_height = original_size
            image = Image.open(io.BytesIO(png_bytes))
        elif mime_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_content))
            original_width, original_height = image.size
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image, PDF, DOCX, or PPTX.")

        png_buffer = io.BytesIO()
        image.convert("RGB").save(png_buffer, format="PNG") # Ensure RGB for consistent processing
        png_buffer.seek(0)
        png_image = Image.open(png_buffer)
        resized_image = resize_image_if_needed(png_image)
        width, height = resized_image.size # Get dimensions after resizing

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe this image?"}
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[resized_image], return_tensors="pt").to(DEVICE)

        generation_start_time = time()
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = generated_texts[0]
        generation_end_time = time()
        generation_time_seconds = generation_end_time - generation_start_time

        end_process_time = time()
        total_time_seconds = end_process_time - start_process_time
        end_datetime = datetime.utcnow().isoformat() + 'Z'

        task_results[task_id] = {
            "id": task_id,
            "object": "file_description_task",
            "created": start_datetime,
            "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "task_status": "completed",
            "description": output_text,
            "processing_details": {
                "start_time": start_datetime,
                "end_time": end_datetime,
                "total_time_seconds": f"{total_time_seconds:.4f}",
                "generation_time_seconds": f"{generation_time_seconds:.4f}",
                "file_metadata": {
                    "filename": filename,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "original_width": original_width,
                    "original_height": original_height,
                    "processed_width": width,
                    "processed_height": height,
                    "processed_format": processed_format
                }
            }
        }
    except HTTPException as http_exc: # Catch HTTPException from file type check
        task_results[task_id] = {
            "id": task_id,
            "object": "file_description_task",
            "created": start_datetime,
            "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "task_status": "error",
            "error_message": http_exc.detail, # Use detail from HTTPException
            "processing_details": {
                "start_time": start_datetime,
                "end_time": datetime.utcnow().isoformat() + 'Z',
                "total_time_seconds": f"{time() - start_process_time:.4f}",
                "file_metadata": {
                    "filename": filename,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "error": http_exc.detail # Indicate error in metadata as well
                }
            }
        }
    except Exception as e: # Catch other exceptions during processing
        end_process_time = time()
        total_time_seconds = end_process_time - start_process_time
        end_datetime = datetime.utcnow().isoformat() + 'Z'
        task_results[task_id] = {
            "id": task_id,
            "object": "file_description_task",
            "created": start_datetime,
            "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "task_status": "error",
            "error_message": str(e),
            "processing_details": {
                "start_time": start_datetime,
                "end_time": end_datetime,
                "total_time_seconds": f"{total_time_seconds:.4f}",
                "file_metadata": {
                    "filename": filename,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "error": "Processing failed" # General error message
                }
            }
        }

@app.post("/describe_file/")
async def describe_file_endpoint(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_request_time = time()
    start_datetime_request = datetime.utcnow().isoformat() + 'Z'
    mime_type = file.content_type
    filename = file.filename

    allowed_mime_types = ["image/png", "image/jpeg", "image/webp", "application/pdf",
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                          "application/vnd.openxmlformats-officedocument.presentationml.presentation"]

    if mime_type not in allowed_mime_types and not mime_type.startswith("image/"): # To broadly allow all image types if some new image mime is used
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image, PDF, DOCX, or PPTX.")

    task_id = str(uuid.uuid4())
    task_results[task_id] = {
        "id": task_id,
        "object": "file_description_task",
        "created": start_datetime_request,
        "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "task_status": "processing",
    } # Initial status

    file_content = await file.read()
    size_bytes = len(file_content)

    background_tasks.add_task(
        process_image_description,
        file_content,
        task_id,
        filename,
        mime_type,
        size_bytes
    )

    end_request_time = time()
    total_request_time = end_request_time - start_request_time

    initial_response_payload = {
        "id": task_id,
        "object": "file_description_task",
        "created": start_datetime_request,
        "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "task_status": "processing",
        "message": "File processing started. Check task status using /task_status/{task_id}",
        "processing_details": {
            "request_received_time": start_datetime_request,
            "initial_response_time": datetime.utcnow().isoformat() + 'Z',
            "initial_response_latency_seconds": f"{total_request_time:.4f}",
            "file_metadata": {
                "filename": filename,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
            }
            # "client_ip": request.client.host # You can add this, but consider privacy implications
        }
    }

    return JSONResponse(content=initial_response_payload)

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    result = task_results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=result)

@app.get("/health/")
def health():
    return "VLM is running fine. OCR is ready!"