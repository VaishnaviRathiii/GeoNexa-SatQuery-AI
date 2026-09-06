def classify_question(question: str) -> str:
    q = question.lower().strip()

    if any(word in q for word in ["more", "less", "fewer", "greater", "higher", "lower"]):
        return "comparison"

    if q.startswith(("is ", "are ", "does ", "do ", "can ")):
        return "yes_no"

    if any(phrase in q for phrase in ["how many", "number of", "amount of", "count of"]):
        return "counting"

    if "area" in q:
        return "area"

    return "general"

def build_prompt(question: str) -> str:
    question_type = classify_question(question)

    if question_type == "counting":
        instructions = """
Count the requested objects carefully.
Scan the entire image systematically.
Do not count the same object twice.
Return the final count clearly.
"""

    elif question_type == "yes_no":
        instructions = """
Answer the question with Yes or No first.
Use only information visible in the image.
"""

    elif question_type == "comparison":
        instructions = """
Compare the requested objects or regions carefully.
Base the answer only on visible image evidence.
"""

    elif question_type == "area":
        instructions = """
Analyze the requested area using the available image information.
Do not invent measurements that cannot be determined from the image.
"""

    else:
        instructions = """
Analyze the satellite image carefully and answer the question using visible evidence.
"""

    return f"""
You are GeoNexa, a remote-sensing image analysis assistant.

Question:
{question}

Instructions:
{instructions}

Give a concise and accurate answer.
"""