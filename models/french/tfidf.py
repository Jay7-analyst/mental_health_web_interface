import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


def patch_numpy_pickle_compatibility():
    """
    Juliette's TF-IDF pickle was saved with a NumPy version that references
    numpy._core.numeric. This patch maps that path to the NumPy version
    installed on this machine.
    """
    try:
        import numpy
        import numpy.core
        import numpy.core.numeric
        import numpy.core.multiarray
        import numpy.core.umath

        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.numeric"] = numpy.core.numeric
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
        sys.modules["numpy._core.umath"] = numpy.core.umath
    except Exception:
        pass


class TfidfVectorizerWrapper:

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.vectorizer   = TfidfVectorizer(max_features=max_features,
                                            ngram_range=ngram_range)
        self._is_fitted   = False

    def fit_transform(self, texts):
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        X = X.toarray() if issparse(X) else X
        print(f"[TF-IDF] fit_transform → {X.shape}")
        return X

    def transform(self, texts):
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        X = self.vectorizer.transform(texts)
        X = X.toarray() if issparse(X) else X
        print(f"[TF-IDF] transform → {X.shape}")
        return X

    def transform_one(self, text):
        return self.transform([text])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[TF-IDF] Saved → {path}")

    @classmethod
    def load(cls, path):
        patch_numpy_pickle_compatibility()

        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"[TF-IDF] Loaded ← {path}")
        return obj