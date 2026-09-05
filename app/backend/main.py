"""
GeoNexa Backend (M4's area - Mihir).

Basic architecture:
    Frontend -> Backend -> AI / CV / Geo -> Combined Result -> Frontend

This is the 40%-milestone version: one endpoint, mocked module outputs
that will be swapped for the real AI/CV/Geo functions as teammates finish
their work. Keep this simple - no complicated backend needed yet.

Run locally with:
    uvicorn app.backend.main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

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
    question: str = Form(...),
):
    """
    Main endpoint for the 40% milestone demo.

    Input:  an uploaded image + a text question (multipart/form-data)
    Output: combined result from AI, CV and Geo modules
    """
    image_bytes = await image.read()

    # Call each module - currently mocked, will be swapped for real
    # implementations as M1, M2, M3 finish their work.
    answer = get_ai_answer(image_bytes, question)
    cv_result = get_cv_analysis(image_bytes)
    geo_result = get_geo_info(image_bytes)

    return AnalyzeResponse(
        answer=answer,
        cv_analysis=cv_result,
        geo_info=geo_result,
    )