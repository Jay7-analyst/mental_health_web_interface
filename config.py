# Global configuration

# Keep this True while testing the interface without real models.
# Set to False after placing the real model files and adapters are ready.
DEMO_MODE = False

SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
}

# Labels are language-specific.
# Do not hardcode one global category list here.
# Each adapter returns its own labels/probabilities when available.
