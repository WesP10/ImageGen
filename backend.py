from diffusers import StableDiffusionXLPipeline
import torch
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_file
import sys
import traceback

def load_models():
    try:
        print("Starting to load AI model... This may take a few minutes...")
        
        print("Loading text-to-image model...")
        # Load only the text-to-image model for now
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to("cpu")
        print("✓ Text-to-image model loaded successfully")
        
        return pipe
    except Exception as e:
        print("Error during model loading:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

try:
    print("Initializing server...")
    # Load only the text-to-image model
    pipe = load_models()

    # Create Flask app after model is loaded
    app = Flask(__name__)

    @app.route('/generate/<prompt>', methods=['POST'])
    def generate_image(prompt):
        try:
            print(f"Generating image for prompt: {prompt}")
            # Generate the image
            image = pipe(prompt).images[0]

            # Save image to a byte stream
            img_io = BytesIO()
            image.save(img_io, format="PNG")
            img_io.seek(0)

            print("Image generated successfully")
            return send_file(
                img_io,
                mimetype="image/png",
                as_attachment=True,
                download_name="generated_image.png"
            )
        except Exception as e:
            print(f"Error generating image: {str(e)}", file=sys.stderr)
            return jsonify({"error": str(e)}), 500

    if __name__ == '__main__':
        print("\nModel loaded successfully!")
        print("Starting Flask server on http://localhost:8080")
        app.run(host='localhost', port=8080, debug=False)

except Exception as e:
    print("Fatal error during server initialization:")
    print(str(e))
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)