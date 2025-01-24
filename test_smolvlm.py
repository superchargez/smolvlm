from time import time; start = time()
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
import_time = time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
# image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
# image = load_image(r"/mnt/c/Users/Administrator/Desktop/errored prompts.png")
# image = load_image(r"/mnt/c/Users/Administrator/Documents/data/all/qwen2-vl.jpeg")
image = load_image(r"/mnt/c/Users/Administrator/Documents/data/all/rabbit.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    # quantization_config=quantization_config,
    # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
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
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)
image_processing_time = time()
# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
output_generation_time = time()
print(generated_texts[0])
print(f"OUTPUT GENERATION TIME 1: {output_generation_time - image_processing_time}")
print(f"Model Importing Time: {output_generation_time - model_loading_time}")
print(f"Setting up time: {model_loading_time - start}")
print(f"Total time: {output_generation_time - start}")
