# Multilingual Mental Health Classifier Web Interface

This project is prepared as a Streamlit prototype while waiting for the trained model files.

## Current mode

The app currently runs in demo mode:

```python
DEMO_MODE = True
```

Demo mode does not load real models. It only shows sample outputs so the interface can be tested.

## Final mode

After the Arabic, English, and French model files arrive:

1. Place each model package inside the correct folder:
   - `models/arabic/`
   - `models/english/`
   - `models/french/`

2. Update the matching adapter:
   - `adapters/arabic_adapter.py`
   - `adapters/english_adapter.py`
   - `adapters/french_adapter.py`

3. Set:

```python
DEMO_MODE = False
```

inside `config.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required files from each model owner

Each language should ideally include:
- Saved trained model file
- Saved fitted vectorizer/tokenizer file
- Label mapping or class order
- Preprocessing code
- Requirements/library versions
- One example input and output
