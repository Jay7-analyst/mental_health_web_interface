from inference.output_format import PredictionResult


def predict_demo(text: str, detected_language: str) -> PredictionResult:
    """Temporary demo prediction used while waiting for real model files."""
    selected_model = f"{detected_language} Classifier"

    return PredictionResult(
        detected_language=detected_language,
        selected_model=selected_model,
        predicted_category="Anxiety / Fear",
        confidence_score=0.86,
        class_probabilities={
            "Anxiety / Fear": 0.86,
            "Depression": 0.09,
            "OCD / Obsessive": 0.05,
        },
        mode="demo",
    )
