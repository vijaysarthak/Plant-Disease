"""
api.py — the BACKEND (REST API) for other programs, e.g. a mobile app.
It uses the SAME predictor and database as the Streamlit page.

Run it from the project folder:
    uvicorn app.api:app --reload
Then open http://localhost:8000/docs — FastAPI makes a page where you can try every endpoint.

Endpoints:
    GET  /health        is the server running? which model is loaded?
    POST /predict       upload a leaf photo -> decision + advice (saved in the database)
    POST /feedback      was a prediction right?
    GET  /history       the latest predictions
"""
import io
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src import database
from src.predictor import PlantDiseasePredictor

app = FastAPI(title="Plant Disease Advisor API")
predictor = PlantDiseasePredictor()        # loaded once, when the server starts
database.create_tables()


class Feedback(BaseModel):
    """What the client sends to /feedback."""
    prediction_id: int
    is_correct: bool
    true_class: str = ""
    comment: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.info["backbone"], "trained_on": predictor.info["trained_on"]}


@app.post("/predict")
async def predict(photo: UploadFile = File(...)):
    photo_bytes = await photo.read()
    try:
        image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The file is not an image.")
    result = predictor.predict(image)
    result["prediction_id"] = database.save_prediction(photo.filename, result)
    return result


@app.post("/feedback")
def feedback(item: Feedback):
    database.save_feedback(item.prediction_id, item.is_correct, item.true_class, item.comment)
    return {"saved": True}


@app.get("/history")
def history(limit: int = 20):
    return database.get_history(limit)
