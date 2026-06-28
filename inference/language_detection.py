import re

from langdetect import detect, LangDetectException
from config import SUPPORTED_LANGUAGES


ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")
LATIN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")

# French-specific clues only.
# Do NOT include generic words like depression/depression because English can use them too.
FRENCH_HINT_PATTERN = re.compile(
    r"\b("
    r"je|j'ai|j’ai|jme|moi|suis|être|etre|"
    r"triste|angoissé|angoissee|angoissée|anxieux|anxieuse|"
    r"peur|dormir|aide|besoin|mal|souffre|depuis|"
    r"bonjour|merci|docteur"
    r")\b",
    re.IGNORECASE,
)

FRENCH_ACCENT_PATTERN = re.compile(r"[àâçéèêëîïôûùüÿœæ]", re.IGNORECASE)


def detect_language_name(text: str) -> tuple[str, str]:
    """Return (language_code, language_name)."""
    text = str(text).strip()

    if not text:
        return "unsupported", "Unsupported"

    # Arabic is direct and reliable.
    if ARABIC_PATTERN.search(text):
        return "ar", "Arabic"

    # Latin-script input: decide between French and English.
    if LATIN_PATTERN.search(text):
        lowered = text.lower()

        # French only when there are French-specific accents or clear French words.
        if FRENCH_ACCENT_PATTERN.search(lowered) or FRENCH_HINT_PATTERN.search(lowered):
            return "fr", "French"

        # Default Latin text to English.
        return "en", "English"

    # Fallback for anything not caught above.
    try:
        language_code = detect(text)
    except LangDetectException:
        return "unsupported", "Unsupported"

    language_name = SUPPORTED_LANGUAGES.get(language_code, "Unsupported")
    return language_code, language_name