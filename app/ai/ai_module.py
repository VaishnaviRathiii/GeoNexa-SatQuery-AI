"""
AI / VLM module (M1's area).

This is a MOCK implementation. It exists so the backend can be built and
tested end-to-end before the real VLM pipeline is ready. Once M1's real
function is ready, replace the body of get_ai_answer() with the real call,
keeping the same input/output shape:

    input:  image (bytes) + question (str)
    output: answer (str)
"""


def get_ai_answer(image_bytes: bytes, question: str) -> str:
    """
    Mock AI answer generator.
    Real version (M1) will run the VLM on the image + question and
    return a generated text answer.
    """
    return f"[MOCK AI ANSWER] Based on the image, here's a placeholder answer to: '{question}'"