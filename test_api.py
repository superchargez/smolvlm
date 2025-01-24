from time import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
import io

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

def resize_image_if_needed(image):
    """
    Resizes an image if it's larger than 512x512 pixels, maintaining aspect ratio
    and aiming for optimal quality for the SmolVLM model.

    Args:
        image: PIL Image object

    Returns:
        PIL Image object, resized if necessary
    """
    width, height = image.size
    if width <= 512 and height <= 512:
        return image  # No resizing needed

    if width > height:
        new_width = 512
        new_height = int(height * (512 / width))
    else:
        new_height = 512
        new_width = int(width * (512 / height))

    resized_image_np = cv2.resize(np.array(image), (new_width, new_height), interpolation=cv2.INTER_AREA) # Using INTER_AREA for shrinking
    resized_image_pil = Image.fromarray(resized_image_np)
    return resized_image_pil

@app.post("/describe_image/")
async def describe_image_endpoint(image_file: UploadFile = File(...)):
    start_time = time()
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_content = await image_file.read()
        image = Image.open(io.BytesIO(image_content))

        # Convert image to PNG for best OCR and detail preservation
        png_buffer = io.BytesIO()
        image.convert("RGB").save(png_buffer, format="PNG") # Convert to RGB first to handle potential palette issues
        png_buffer.seek(0) # Reset buffer to beginning to read from it
        png_image = Image.open(png_buffer) # Re-open as PNG

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

        generation_start_time = time()
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = generated_texts[0]
        generation_end_time = time()

        total_time = time() - start_time
        generation_time = generation_end_time - generation_start_time

        return JSONResponse(content={
            "description": output_text,
            "processing_time_seconds": f"{total_time:.4f}",
            "generation_time_seconds": f"{generation_time:.4f}",
            "image_mime_type_received": image_file.content_type,
            "image_format_processed": "PNG" # Indicate that image was processed as PNG
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health/")
def health():
    return "VLM is running fine!"