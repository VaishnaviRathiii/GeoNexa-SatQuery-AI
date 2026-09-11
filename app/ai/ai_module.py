"""
AI / VLM module (M1's area).

Connects the backend image bytes + question
to the real M1 VLM pipeline.
"""

from io import BytesIO

from PIL import Image

from app.ai.model import load_model
from app.ai.pipeline import analyze_image


# Load the model once and reuse it
_model = None
_processor = None
_device = None


def get_ai_model():
    """Load the VLM only when it is first needed."""
    global _model, _processor, _device

    if _model is None or _processor is None:
        _model, _processor, _device = load_model()

    return _model, _processor, _device


def get_ai_answer(image_bytes: bytes, question: str) -> str:
    """
    Run the real M1 VLM pipeline.

    Input:
        image_bytes: uploaded image as bytes
        question: user's natural-language question

    Output:
        generated AI answer as a string
    """

    try:
        # Convert uploaded bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Load/reuse Qwen model
        model, processor, device = get_ai_model()

        # Run M1 pipeline
        result = analyze_image(
            question=question,
            image=image,
            model=model,
            processor=processor,
            device=device,
        )

        return result["answer"]

    except Exception as e:
        return f"AI analysis failed: {str(e)}"