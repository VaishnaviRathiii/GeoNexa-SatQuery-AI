"""
AI / VLM module (M1's area).

This is a compatible interface implementation so backend and CV pipelines
can interact cohesively across development branches.
"""


def get_ai_answer(image_bytes: bytes, question: str) -> str:
    """
    Mock AI answer generator.
    Real version (M1) runs VLM on image + question.
    """
    return f"[MOCK AI ANSWER] Based on the image, here is a placeholder answer to: '{question}'"
