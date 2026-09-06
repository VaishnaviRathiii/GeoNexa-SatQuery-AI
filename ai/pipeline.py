from ai.query import build_prompt


def prepare_query(question: str) -> str:
    return build_prompt(question)


def analyze_image(question: str, image=None, model=None, processor=None):
    if model is None or processor is None:
        return {
            "question": question,
            "answer": None,
            "status": "model_not_loaded"
        }

    prompt = prepare_query(question)

    return {
        "question": question,
        "prompt": prompt,
        "answer": None,
        "status": "ready_for_inference"
    }