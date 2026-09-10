"""
GeoNexa Backend (M4's area - Mihir).

Basic architecture:
    Frontend -> Backend -> AI / CV / Geo -> Combined Result -> Frontend
"""

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

from app.ai.ai_module import get_ai_answer
from app.computer_vision.cv_module import get_cv_analysis
from app.geospatial.geo_module import get_geo_info

app = FastAPI(title="GeoNexa Backend", version="0.1.0")


class AnalyzeResponse(BaseModel):
    """Shape of the combined result sent back to the frontend."""
    answer: str
    cv_analysis: dict
    geo_info: dict


@app.get("/")
def root():
    """Simple health check so you can confirm the server is running."""
    return {"status": "GeoNexa backend is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    question: Optional[str] = Form(""),
):
    """
    Main endpoint for GeoNexa image analysis.

    Input:  an uploaded image + a text question (multipart/form-data)
    Output: combined result from AI, CV and Geo modules
    """
    image_bytes = await image.read()

    answer = get_ai_answer(image_bytes, question or "")
    cv_result = get_cv_analysis(image_bytes, query=question)
    geo_result = get_geo_info(image_bytes)

    return AnalyzeResponse(
        answer=answer,
        cv_analysis=cv_result,
        geo_info=geo_result,
    )
