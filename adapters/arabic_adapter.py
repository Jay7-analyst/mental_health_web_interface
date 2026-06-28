from pathlib import Path
import sys
import pickle
import json

import numpy as np
import streamlit as st

from inference.output_format import PredictionResult


MODEL_FOLDER = Path("models/arabic")
VECTORIZER_PATH = MODEL_FOLDER / "arabic_tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_FOLDER / "arabic_logistic_regression.pkl"
LABEL_ENCODER_PATH = MODEL_FOLDER / "arabic_label_encoder.pkl"
CONFIG_PATH = MODEL_FOLDER / "arabic_config.json"

# Arabic preprocessing lives inside models/arabic.
arabic_folder = str(MODEL_FOLDER.resolve())
if arabic_folder not in sys.path:
    sys.path.insert(0, arabic_folder)

from arabic_preprocessing import ArabicPreprocessor


@st.cache_resource(show_spinner=False)
def load_arabic_resources():
    missing_files = []

    for path in [VECTORIZER_PATH, MODEL_PATH, LABEL_ENCODER_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Missing Arabic model files:\n" + "\n".join(missing_files)
        )

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    preprocessor = ArabicPreprocessor()

    return vectorizer, model, label_encoder, config, preprocessor


def predict(text: str, detected_language: str) -> PredictionResult:
    vectorizer, model, label_encoder, config, preprocessor = load_arabic_resources()

    analysis_text = preprocessor.preprocess_for_model(text)

    if not analysis_text.strip():
        return PredictionResult(
            detected_language=detected_language,
            selected_model="Arabic TF-IDF + Logistic Regression",
            predicted_category="Unable to classify",
            confidence_score=None,
            confidence_available=False,
            class_probabilities={},
            mode="real",
            decision_strength=None,
            note="The Arabic patient input became empty after EDA-derived preprocessing.",
        )

    X = vectorizer.transform([analysis_text])

    prediction = model.predict(X)
    predicted_category = str(label_encoder.inverse_transform(prediction)[0])

    confidence_score = None
    confidence_available = False
    class_probabilities = {}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = label_encoder.inverse_transform(model.classes_)

        class_probabilities = {
            str(class_name): float(prob)
            for class_name, prob in zip(classes, probs)
        }

        confidence_score = float(np.max(probs))
        confidence_available = True

    return PredictionResult(
        detected_language=detected_language,
        selected_model="Arabic TF-IDF + Logistic Regression",
        predicted_category=predicted_category,
        confidence_score=confidence_score,
        confidence_available=confidence_available,
        class_probabilities=class_probabilities,
        mode="real",
        decision_strength=None,
        note=(
            "Arabic prediction uses EDA-derived preprocessing, then TF-IDF + Logistic Regression. "
            f"analysis_text: {analysis_text}"
        ),
    )