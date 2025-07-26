Controllable Image Generation for Architectural Design

This project demonstrates a state-of-the-art computer vision pipeline for controllable image generation. It uses a combination of a powerful text-to-image model (Stable Diffusion) and a conditional control model (ControlNet) to generate realistic architectural images that adhere to both a text description and a user-provided structural sketch.

This is an advanced application of generative AI that goes beyond simple text prompts to give users fine-grained control over the creative output.
Methodology: Stable Diffusion + ControlNet

The core of this project is the ControlNet architecture, which acts as a "harness" for large diffusion models like Stable Diffusion.

    Stable Diffusion (The Creative Engine): A powerful, pre-trained model that can generate high-quality images from text prompts alone. However, it lacks precise control over the structure and composition of the output.

    ControlNet (The Structural Guide): A secondary neural network that is trained to understand specific types of conditional inputs, such as edge maps, human poses, or depth maps. It provides a powerful way to guide the image generation process.

    The Pipeline:

        A user provides two inputs: a text prompt (e.g., "a modern brick house") and a structural image (e.g., a simple architectural sketch).

        The structural image is preprocessed using an algorithm like Canny edge detection to create a clean "edge map."

        Both the text prompt and the edge map are fed into the combined Stable Diffusion + ControlNet pipeline.

        Stable Diffusion generates the image based on the creative description in the text, but ControlNet forces the generation process to strictly follow the lines and shapes present in the edge map.

The result is an image that has the artistic quality from the text prompt and the exact composition from the sketch.
Project Structure

This repository contains a single, comprehensive Python script:

    generate_architectural_image.py: This script simulates the entire end-to-end pipeline.

        Configuration: Defines the Hugging Face model IDs for Stable Diffusion and the Canny Edge ControlNet.

        Image Preprocessing: Contains a function that uses OpenCV to convert a sketch into a Canny edge map suitable for ControlNet.

        Generative Pipeline: Defines the main function that (simulates) loading the models using the Hugging Face diffusers library and running the generation process with both the text prompt and the control image as inputs.

        Execution: The main block demonstrates the full workflow, from processing a sketch to generating the final design.

How to Run
Prerequisites

You will need a Python environment with a powerful GPU (NVIDIA, with at least 12GB of VRAM) and the following libraries installed:

pip install torch diffusers transformers accelerate opencv-python Pillow

Execution

    (Optional) Create a simple black-and-white sketch of a building and save it as a .png file.

    Update the user_sketch_path and user_text_prompt variables in the generate_architectural_image.py script.

    Run the script from your terminal:

    python generate_

