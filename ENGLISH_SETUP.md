# English model setup

The English adapter is now prepared for Joseph's SVM + DistilBERT model.

## Files to copy

Copy these into `models/english/`:

- `svm_distilbert.pkl`
- `vectorizers.py`
- `cleaning.py`
- `config.py`
- full `ml/` folder from Joseph

## Then edit config.py

Change:

```python
DEMO_MODE = True
```

to:

```python
DEMO_MODE = False
```

## Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Limitation

English confidence is not available because the saved SVM does not support `predict_proba`.
The app can show decision strength instead, but this is not a real probability.
