from config import DEMO_MODE
from inference.language_detection import detect_language_name
from adapters.demo_adapter import predict_demo


def predict_text(text: str):
    language_code, language_name = detect_language_name(text)

    if language_name == "Unsupported":
        return {
            "ok": False,
            "error": "Unsupported language. Please enter Arabic, English, or French text.",
        }

    if DEMO_MODE:
        return {
            "ok": True,
            "result": predict_demo(text=text, detected_language=language_name),
        }

    if language_code == "ar":
        from adapters.arabic_adapter import predict
    elif language_code == "en":
        from adapters.english_adapter import predict
    elif language_code == "fr":
        from adapters.french_adapter import predict
    else:
        return {
            "ok": False,
            "error": "Unsupported language. Please enter Arabic, English, or French text.",
        }

    return {
        "ok": True,
        "result": predict(text=text, detected_language=language_name),
    }
