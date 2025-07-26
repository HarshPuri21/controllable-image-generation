# generate_architectural_image.py
# This script demonstrates a controllable image generation pipeline using
# Stable Diffusion and ControlNet to create architectural designs from
# text prompts and structural sketches.

import torch
from PIL import Image
import cv2 # OpenCV for image processing
import numpy as np
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# --- 1. Configuration ---

# Base model for image generation (Stable Diffusion)
base_model_id = "runwayml/stable-diffusion-v1-5"
# ControlNet model specialized in understanding edges (sketches)
controlnet_model_id = "lllyasviel/sd-controlnet-canny"

# --- 2. Image Preprocessing with OpenCV ---

def process_sketch_for_controlnet(input_image_path):
    """
    Takes a path to a sketch image, processes it using OpenCV's Canny edge
    detection, and returns an image ready for ControlNet.
    """
    print(f"Processing sketch from: {input_image_path}")
    
    # In a real scenario, you would load the image like this:
    # image = Image.open(input_image_path)
    # image = np.array(image)
    
    # For demonstration, create a mock sketch image (a house outline)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image.fill(255) # White background
    # Draw a simple house shape
    cv2.rectangle(image, (100, 250), (400, 500), (0, 0, 0), 3) # Base
    cv2.line(image, (100, 250), (250, 100), (0, 0, 0), 3) # Roof left
    cv2.line(image, (400, 250), (250, 100), (0, 0, 0), 3) # Roof right
    cv2.rectangle(image, (150, 350), (220, 500), (0, 0, 0), 3) # Door
    cv2.rectangle(image, (300, 350), (350, 400), (0, 0, 0), 3) # Window
    
    # Apply Canny edge detection
    # This finds the sharp edges in the image, which is what ControlNet uses
    # as a structural guide.
    low_threshold = 100
    high_threshold = 200
    canny_edges = cv2.Canny(image, low_threshold, high_threshold)
    
    # The Canny output is a 2D array, but ControlNet expects a 3D image.
    # We stack the edges into three channels.
    canny_edges = canny_edges[:, :, None]
    canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
    
    # Convert back to a PIL Image
    control_image = Image.fromarray(canny_edges)
    
    print("Sketch processed into Canny edge map.")
    control_image.save("canny_edge_map.png")
    print("Edge map saved as 'canny_edge_map.png'")
    
    return control_image

# --- 3. The Generative Pipeline ---

def generate_controlled_image(prompt, control_image):
    """
    Loads the Stable Diffusion and ControlNet models and generates an image
    that adheres to both the text prompt and the control image.
    """
    print("\n--- Initializing Generative Pipeline ---")
    
    # In a real application, you would load the models from the Hugging Face Hub.
    # This requires significant VRAM and download time.
    
    # print("Loading ControlNet model...")
    # controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
    
    # print("Loading Stable Diffusion pipeline...")
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     base_model_id,
    #     controlnet=controlnet,
    #     torch_dtype=torch.float16,
    #     safety_checker=None # Disable for performance if not needed
    # ).to("cuda") # Move the pipeline to the GPU
    
    # # Use an efficient scheduler
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    print("--- SIMULATING MODEL LOADING ---")
    print(f"Base Model: {base_model_id}")
    print(f"ControlNet Model: {controlnet_model_id}")
    print("Models would be loaded onto the GPU here.")
    
    print("\n--- Starting Image Generation ---")
    print(f"Text Prompt: '{prompt}'")
    print("ControlNet is using the Canny edge map as a structural guide.")
    
    # Generate the image
    # In a real scenario, you would run this line:
    # result_image = pipe(
    #     prompt,
    #     num_inference_steps=30,
    #     generator=torch.manual_seed(0),
    #     image=control_image
    # ).images[0]
    
    print("--- SIMULATING IMAGE GENERATION ---")
    # For this demonstration, we'll just return the control image itself
    # to show what the structural guide looks like.
    result_image = control_image.point(lambda p: 255 if p > 0 else 0) # Make lines solid black
    
    print("Image generation complete.")
    result_image.save("generated_architectural_design.png")
    print("Final image saved as 'generated_architectural_design.png'")
    
    return result_image

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Define the user's creative input
    user_sketch_path = "path/to/my_house_sketch.png"
    user_text_prompt = "A hyperrealistic photo of a modern brick house with large, glowing windows, surrounded by a lush forest at sunset."

    # 1. Process the sketch
    control_image = process_sketch_for_controlnet(user_sketch_path)

    # 2. Generate the final image
    final_design = generate_controlled_image(user_text_prompt, control_image)

    print("\nProject execution finished. Check the saved .png files.")

