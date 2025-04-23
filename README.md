# Image Generation Project

This project provides a simple interface for generating images using a locally-hosted Stable Diffusion model. It allows users to input text prompts or upload images, and the model processes these inputs to generate creative outputs.

## Features
- Accepts text prompts for image generation.
- Supports image uploads for processing.
- Runs a locally-hosted Stable Diffusion model to ensure privacy and control.

## Requirements
- Python 3.8 or newer
- Compatible GPU with CUDA support (recommended for faster performance)
- Required Python libraries:
  - `torch`
  - `diffusers`
  - `flask`
  - `pillow`
  - Additional dependencies listed in `requirements.txt`

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   pip install -r requirements.txt
   python app.py