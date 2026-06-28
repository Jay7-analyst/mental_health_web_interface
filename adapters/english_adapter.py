from pathlib import Path
import sys
import pickle

import joblib
import numpy as np
import streamlit as st

from inference.output_format import PredictionResult


MODEL_FOLDER = Path("models/english")

OLD_MODEL_PATH = MODEL_FOLDER / "svm_distilbert.pkl"

PROBABILITY_MODEL_PATH = MODEL_FOLDER / "english_svm_distilbert_probability.pkl"
SCALER_PATH = MODEL_FOLDER / "english_distilbert_scaler.pkl"

# English model dependencies live inside models/english.
english_folder = str(MODEL_FOLDER.resolve())
if english_folder not in sys.path:
    sys.path.insert(0, english_folder)


def _load_pickle(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_english_resources():
    # These imports must happen after models/english is added to sys.path.
    from cleaning import Cleaning
    from vectorizers import generate_distilbert_embeddings

    cleaner = Cleaning()

    # Preferred path: Joseph's probability-enabled model.
    if PROBABILITY_MODEL_PATH.exists() and SCALER_PATH.exists():
        probability_model = _load_pickle(PROBABILITY_MODEL_PATH)
        scaler = _load_pickle(SCALER_PATH)

        return {
            "mode": "probability",
            "model": probability_model,
            "scaler": scaler,
            "cleaner": cleaner,
            "embedder": generate_distilbert_embeddings,
        }

    # Backup path: old English model.
    if OLD_MODEL_PATH.exists():
        old_model_wrapper = joblib.load(OLD_MODEL_PATH)

        return {
            "mode": "old",
            "model": old_model_wrapper,
            "scaler": None,
            "cleaner": cleaner,
            "embedder": generate_distilbert_embeddings,
        }

    raise FileNotFoundError(
        "Missing English model files. Expected either:\n"
        "- models/english/english_svm_distilbert_probability.pkl and models/english/english_distilbert_scaler.pkl\n"
        "or\n"
        "- models/english/svm_distilbert.pkl"
    )


def _decision_strength(model_wrapper, X) -> str | None:
    internal_model = getattr(model_wrapper, "model", model_wrapper)

    if not hasattr(internal_model, "decision_function"):
        return None

    try:
        scores = internal_model.decision_function(X)
        scores = np.asarray(scores)

        margin = float(np.max(np.abs(scores)))

        if margin >= 2.0:
            return "Strong"
        if margin >= 1.0:
            return "Medium"
        return "Weak"
    except Exception:
        return None


def predict(text: str, detected_language: str) -> PredictionResult:
    resources = load_english_resources()

    model = resources["model"]
    scaler = resources["scaler"]
    cleaner = resources["cleaner"]
    generate_distilbert_embeddings = resources["embedder"]

    cleaned_text = cleaner.full_preprocess(text)
    X = generate_distilbert_embeddings([cleaned_text])

    # New probability-enabled path.
    if resources["mode"] == "probability":
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        predicted_label = str(prediction[0])

        confidence_score = None
        confidence_available = False
        class_probabilities = {}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            class_probabilities = {
                str(class_name): float(prob)
                for class_name, prob in zip(model.classes_, probs)
            }

            confidence_score = float(np.max(probs))
            confidence_available = True

        decision_strength = _decision_strength(model, X_scaled)

        return PredictionResult(
            detected_language=detected_language,
            selected_model="English SVM + DistilBERT",
            predicted_category=predicted_label,
            confidence_score=confidence_score,
            confidence_available=confidence_available,
            class_probabilities=class_probabilities,
            mode="real",
            decision_strength=decision_strength,
            note=(
                "English prediction uses DistilBERT embeddings, the saved scaler, "
                "and Joseph's probability-enabled SVM."
            ),
        )

    # Old fallback path.
    prediction = model.predict(X)
    predicted_label = str(prediction[0])

    decision_strength = _decision_strength(model, X)

    return PredictionResult(
        detected_language=detected_language,
        selected_model="English SVM + DistilBERT",
        predicted_category=predicted_label,
        confidence_score=None,
        confidence_available=False,
        class_probabilities={},
        mode="real",
        decision_strength=decision_strength,
        note=(
            "Fallback English SVM was used. It was saved without probability output, "
            "so confidence percentages are not available."
        ),
    )