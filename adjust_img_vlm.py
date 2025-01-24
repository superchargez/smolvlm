from time import time; start = time()
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
import_time = time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import cv2
import numpy as np
from PIL import Image

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

# Load images
# image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image_path = r"/mnt/c/Users/Administrator/Documents/data/all/rabbit.jpg" # Please adjust this path if needed

# Load image from URL and then from local path (you can choose either)
# image = load_image(image_url)
image = load_image(image_path)

# Resize the image if necessary
resized_image = resize_image_if_needed(image)

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
quantization_config = BitsAndBytesConfig(load_in_8bit=True) # Consider using quantization for memory saving, but might not be necessary for this model size.
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    # quantization_config=quantization_config, # Uncomment to use quantization
    # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager", # Flash attention can speed up on CUDA with supported GPUs
).to(DEVICE)
model_loading_time = time()

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[resized_image], return_tensors="pt") # Use resized_image here
inputs = inputs.to(DEVICE)
image_processing_time = time()

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
output_generation_time = time()

print("Generated Text:", generated_texts[0])
print(f"OUTPUT GENERATION TIME: {output_generation_time - image_processing_time:.4f} seconds")
print(f"Model Importing Time: {output_generation_time - model_loading_time:.4f} seconds")
print(f"Setting up time (Import + Model Load): {model_loading_time - start:.4f} seconds")
print(f"Total time: {output_generation_time - start:.4f} seconds")