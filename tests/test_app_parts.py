"""
Tests for the parts the app uses: image preparation, database, predictor and API.
The predictor/API tests are skipped if no model has been saved yet (run the notebook first).
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import database
from src.image_utils import prepare_image

model_exists = config.MODEL_FILE.exists() and config.MODEL_INFO_FILE.exists()


def make_test_photo():
    pixels = (np.random.default_rng(0).random((300, 400, 3)) * 255).astype("uint8")
    return Image.fromarray(pixels)


def test_prepare_image_shape():
    pixels = prepare_image(make_test_photo())
    assert pixels.shape == (3, 224, 224)
    assert pixels.dtype == np.float32


def test_database_save_and_read(tmp_path):
    db_file = tmp_path / "test.db"
    database.create_tables(db_file)
    fake_result = {"top_classes": [{"class_name": "Tomato___healthy", "probability": 0.9}],
                   "decision": "advise", "advised_group": "Healthy", "prediction_set": ["Healthy"], "energy": -10.0}
    prediction_id = database.save_prediction("leaf.jpg", fake_result, db_file)
    database.save_feedback(prediction_id, True, database_file=db_file)
    history = database.get_history(database_file=db_file)
    assert history[0]["decision"] == "advise"
    assert history[0]["is_correct"] == 1


@pytest.mark.skipif(not model_exists, reason="no saved model yet - run notebooks/train_model.ipynb")
def test_predictor_gives_a_decision():
    from src.predictor import PlantDiseasePredictor
    result = PlantDiseasePredictor().predict(make_test_photo())
    assert result["decision"] in ("advise", "refer", "unknown")
    assert len(result["top_classes"]) == 3
    assert abs(sum(result["group_probabilities"].values()) - 1) < 1e-6


@pytest.mark.skipif(not model_exists, reason="no saved model yet - run notebooks/train_model.ipynb")
def test_api_predict(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_FILE", tmp_path / "api.db")
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    buffer = io.BytesIO()
    make_test_photo().save(buffer, format="JPEG")
    response = client.post("/predict", files={"photo": ("leaf.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["decision"] in ("advise", "refer", "unknown")
