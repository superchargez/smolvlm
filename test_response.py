from time import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
import io
import uuid

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

# In-memory storage for task status and results (for simplicity, use a dictionary)
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

def process_image_description(image_content, task_id):
    """
    Processes the image to generate a description and stores the result.
    This function runs in the background.
    """
    try:
        image = Image.open(io.BytesIO(image_content))
        png_buffer = io.BytesIO()
        image.convert("RGB").save(png_buffer, format="PNG")
        png_buffer.seek(0)
        png_image = Image.open(png_buffer)
        resized_image = resize_image_if_needed(png_image)

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

        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = generated_texts[0]

        task_results[task_id] = {"status": "completed", "description": output_text} # Store result
    except Exception as e:
        task_results[task_id] = {"status": "error", "error_message": str(e)} # Store error

@app.post("/describe_image/")
async def describe_image_endpoint(background_tasks: BackgroundTasks, image_file: UploadFile = File(...)):
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    task_id = str(uuid.uuid4()) # Generate unique task ID
    task_results[task_id] = {"status": "processing"} # Initial status

    image_content = await image_file.read()
    background_tasks.add_task(process_image_description, image_content, task_id) # Run processing in background

    return JSONResponse(content={"message": "Image processing started", "task_id": task_id}) # Immediate response

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    result = task_results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=result)

@app.get("/health/")
def health():
    return "VLM is running fine!"