from pathlib import Path


REQUIRED_EXTENSIONS = {
    "model": [".pkl", ".joblib", ".h5", ".keras", ".pt", ".pth", ".sav"],
    "vectorizer_tokenizer": [".pkl", ".joblib", ".json"],
    "labels": [".json", ".txt", ".csv", ".pkl", ".joblib"],
    "code": [".py", ".ipynb"],
}


def scan_language_folder(language_folder: str | Path) -> dict:
    folder = Path(language_folder)
    files = [p for p in folder.rglob("*") if p.is_file()]

    result = {
        "folder": str(folder),
        "model_files": [],
        "vectorizer_tokenizer_files": [],
        "label_files": [],
        "code_files": [],
        "all_files": [str(p) for p in files],
    }

    for p in files:
        suffix = p.suffix.lower()
        name = p.name.lower()

        if suffix in REQUIRED_EXTENSIONS["model"]:
            result["model_files"].append(str(p))

        if suffix in REQUIRED_EXTENSIONS["vectorizer_tokenizer"] and (
            "vector" in name or "token" in name or "tfidf" in name or "count" in name
        ):
            result["vectorizer_tokenizer_files"].append(str(p))

        if suffix in REQUIRED_EXTENSIONS["labels"] and (
            "label" in name or "class" in name or "encoder" in name or "mapping" in name
        ):
            result["label_files"].append(str(p))

        if suffix in REQUIRED_EXTENSIONS["code"]:
            result["code_files"].append(str(p))

    return result
