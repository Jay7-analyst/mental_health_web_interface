from pathlib import Path
import sys
import os
from contextlib import contextmanager

import streamlit as st

from inference.output_format import PredictionResult


MODEL_FOLDER = Path("models/french").resolve()
CONFIG_PATH = MODEL_FOLDER / "config_exported.yaml"
CHECKPOINT_PATH = MODEL_FOLDER / "checkpoints" / "LSTM_tfidf.pt"
VECTORIZER_PATH = MODEL_FOLDER / "vectorizers" / "tfidf_vectorizer.pkl"

french_folder = str(MODEL_FOLDER)
if french_folder not in sys.path:
    sys.path.insert(0, french_folder)


@contextmanager
def working_directory(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@st.cache_resource(show_spinner=False)
def load_french_resources():
    missing_files = []

    for path in [CONFIG_PATH, CHECKPOINT_PATH, VECTORIZER_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Missing French model files:\n" + "\n".join(missing_files)
        )

    # Juliette's predictor uses relative paths from config_exported.yaml,
    # so load it while the working directory is models/french.
    with working_directory(MODEL_FOLDER):
        from predictor import Predictor
        predictor = Predictor("config_exported.yaml")

    return predictor


def _build_probabilities(label: str, confidence: float | None):
    if confidence is None:
        return {}

    label_lower = label.lower()

    if label_lower == "healthy":
        return {
            "Healthy": confidence,
            "Unhealthy": 1.0 - confidence,
        }

    if label_lower == "unhealthy":
        return {
            "Healthy": 1.0 - confidence,
            "Unhealthy": confidence,
        }

    return {label: confidence}


def predict(text: str, detected_language: str) -> PredictionResult:
    predictor = load_french_resources()

    with working_directory(MODEL_FOLDER):
        output = predictor.predict(text)

    predicted_label = "Unknown"
    confidence_score = None

    if isinstance(output, (tuple, list)):
        if len(output) >= 1:
            predicted_label = str(output[0])
        if len(output) >= 2:
            confidence_score = float(output[1])
    elif isinstance(output, dict):
        predicted_label = str(
            output.get("label")
            or output.get("predicted_label")
            or output.get("prediction")
            or output.get("class")
            or "Unknown"
        )

        confidence_score = (
            output.get("confidence")
            or output.get("confidence_score")
            or output.get("probability")
        )

        if confidence_score is not None:
            confidence_score = float(confidence_score)
    else:
        predicted_label = str(output)

    if confidence_score is not None and confidence_score > 1:
        confidence_score = confidence_score / 100.0

    class_probabilities = _build_probabilities(predicted_label, confidence_score)

    return PredictionResult(
        detected_language=detected_language,
        selected_model="French LSTM + TF-IDF",
        predicted_category=predicted_label,
        confidence_score=confidence_score,
        confidence_available=confidence_score is not None,
        class_probabilities=class_probabilities,
        mode="real",
        decision_strength=None,
        note="French prediction uses Juliette's exported LSTM + TF-IDF model package.",
    )