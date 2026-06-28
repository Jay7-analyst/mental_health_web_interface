import html as html_escape
from pathlib import Path

import streamlit as st

from config import DEMO_MODE
from inference.model_router import predict_text


st.set_page_config(
    page_title="Cross-Lingual Mental Health Classifier",
    layout="wide",
)


def html(content: str):
    cleaned = "\n".join(line.strip() for line in content.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"

    if not css_path.exists():
        st.error(f"CSS file not found: {css_path}")
        return

    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def pretty_label(label: str) -> str:
    mapping = {
        "anxiety_fear": "Anxiety / Fear",
        "depression": "Depression",
        "ocd_obsessive": "OCD / Obsessive",
        "Healthy": "Healthy",
        "Unhealthy": "Unhealthy",
        "Anxiety / Fear": "Anxiety / Fear",
        "Depression": "Depression",
        "OCD / Obsessive": "OCD / Obsessive",
    }
    return mapping.get(str(label), str(label).replace("_", " ").title())


def label_color(label: str) -> str:
    colors = {
        "anxiety_fear": "#FF6B6B",
        "Anxiety / Fear": "#FF6B6B",
        "Anxiety": "#FF6B6B",
        "depression": "#463EBD",
        "Depression": "#463EBD",
        "ocd_obsessive": "#D8BE3D",
        "OCD / Obsessive": "#D8BE3D",
        "Healthy": "#55D7BF",
        "Unhealthy": "#FF6B6B",
        "Normal": "#55D7BF",
        "Suicidal": "#D8BE3D",
    }
    return colors.get(str(label), "#55D7BF")


def clean_model_name(model_name: str) -> str:
    replacements = {
        "Arabic TF-IDF + Logistic Regression": "Logistic Regression + TF-IDF",
        "English SVM + DistilBERT": "SVM + DistilBERT",
        "French LSTM + TF-IDF": "LSTM + TF-IDF",
    }
    return replacements.get(str(model_name), str(model_name))


def confidence_strength(confidence_score, existing_strength=None) -> str | None:
    """
    Decision strength is based on confidence probability for all languages.
    This keeps English, French, and Arabic consistent.
    """
    if confidence_score is None:
        return None

    try:
        confidence = float(confidence_score)
    except Exception:
        return None

    if confidence >= 0.75:
        return "Strong"
    if confidence >= 0.55:
        return "Medium"
    return "Weak"


def strength_class(strength: str | None) -> str:
    if not strength:
        return "unknown"

    value = strength.lower()

    if value == "strong":
        return "strong"
    if value == "medium":
        return "medium"
    if value == "weak":
        return "weak"

    return "unknown"


def get_pipeline_steps(result=None):
    if result is None:
        return [
            ("01", "Input Received", "Waiting for text to be entered."),
            ("02", "Language Detection", "The system detects English, French, or Arabic."),
            ("03", "Model Routing", "The matching classifier is selected."),
            ("04", "Prediction Output", "The result, confidence, and strength are displayed."),
        ]

    language = str(result.detected_language).lower()

    if "arabic" in language:
        return [
            ("01", "Arabic Input", "Text received successfully."),
            ("02", "Preprocessing", "Arabic cleaning pipeline completed."),
            ("03", "TF-IDF Vectorization", "Text transformed into numerical features."),
            ("04", "Logistic Regression", "Prediction completed."),
        ]

    if "english" in language:
        return [
            ("01", "English Input", "Text received successfully."),
            ("02", "Cleaning", "English preprocessing completed."),
            ("03", "DistilBERT Embeddings", "Text represented using DistilBERT features."),
            ("04", "SVM Classification", "Prediction completed."),
        ]

    if "french" in language:
        return [
            ("01", "French Input", "Text received successfully."),
            ("02", "TF-IDF Vectorization", "French text transformed into numerical features."),
            ("03", "LSTM Model", "Deep learning classifier processed the input."),
            ("04", "Prediction Output", "Prediction completed."),
        ]

    return [
        ("01", "Input Received", "Text received successfully."),
        ("02", "Preprocessing", "Text preprocessing completed."),
        ("03", "Feature Extraction", "Features generated successfully."),
        ("04", "Classification", "Prediction completed."),
    ]


def render_pipeline_card(result=None):
    steps = get_pipeline_steps(result)
    is_done = result is not None

    html("""
    <div class="section-card pipeline-card">
        <div class="eyebrow">Methodology Flow</div>
        <div class="section-title">Processing Pipeline</div>
        <div class="section-text">
            The interface shows each inference step from input detection to final classification.
        </div>
        <div class="pipeline-list">
    """)

    for number, title, desc in steps:
        status_class = "done" if is_done else "waiting"
        status_text = "Done" if is_done else "Waiting"

        html(f"""
        <div class="pipeline-step {status_class}">
            <div class="pipeline-number">{html_escape.escape(number)}</div>
            <div class="pipeline-content">
                <div class="pipeline-title-row">
                    <div class="pipeline-title">{html_escape.escape(title)}</div>
                    <div class="pipeline-status">{status_text}</div>
                </div>
                <div class="pipeline-desc">{html_escape.escape(desc)}</div>
            </div>
        </div>
        """)

    html("""
        </div>
    </div>
    """)


def render_empty_result():
    html("""
    <div class="section-card result-dashboard-card">
        <div class="demo-banner muted-banner">Awaiting Input</div>

        <div class="empty-result-center">
            <div class="empty-orb">AI</div>
            <div class="empty-title">Model Result</div>
            <div class="empty-text">
                Submit text to view the detected language, selected model,
                predicted category, confidence score, decision strength, and probability distribution.
            </div>
        </div>

        <div class="placeholder-grid">
            <div class="placeholder-tile">
                <div class="placeholder-label">Detected Language</div>
                <div class="placeholder-value">Waiting</div>
            </div>
            <div class="placeholder-tile">
                <div class="placeholder-label">Prediction</div>
                <div class="placeholder-value">Not analyzed yet</div>
            </div>
        </div>
    </div>
    """)


def render_probability_card(result):
    if not result.class_probabilities:
        html("""
        <div class="section-card probability-card">
            <div class="eyebrow">Probability Output</div>
            <div class="section-title small-title">Class Probability Breakdown</div>
            <div class="no-probability-box">
                <div class="no-probability-title">Probability distribution unavailable</div>
                <div class="no-probability-text">
                    This model does not currently provide class probability values.
                    The predicted category is still displayed in the result card.
                </div>
            </div>
        </div>
        """)
        return

    html("""
    <div class="section-card probability-card">
        <div class="eyebrow">Probability Output</div>
        <div class="section-title small-title">Class Probability Breakdown</div>
        <div class="section-text">
            The bars show how the model distributed confidence across the available categories.
        </div>
    """)

    sorted_probs = sorted(
        result.class_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for label, probability in sorted_probs:
        percent = int(round(float(probability) * 100))
        display_label = pretty_label(label)
        color = label_color(label)

        html(f"""
        <div class="prob-row">
            <div class="prob-title">
                <span>{html_escape.escape(display_label)}</span>
                <span>{percent}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: {percent}%; background: {color};"></div>
            </div>
        </div>
        """)

    html("</div>")


def render_result(result):
    mode_label = "Demo Result" if result.mode == "demo" else "Model Result"

    confidence = result.confidence_percent()
    confidence_display = f"{confidence}%" if result.confidence_available and confidence is not None else "N/A"
    confidence_subtitle = "Confidence" if result.confidence_available else "Unavailable"

    predicted_display = pretty_label(result.predicted_category)
    pill_color = label_color(result.predicted_category)

    selected_model_display = clean_model_name(result.selected_model)

    model_strength = confidence_strength(
        result.confidence_score,
        result.decision_strength,
    )
    model_strength_display = model_strength if model_strength else "N/A"
    model_strength_class = strength_class(model_strength)

    html(f"""
    <div class="section-card result-dashboard-card">
        <div class="demo-banner">{mode_label}</div>

        <div class="result-hero-row">
            <div class="prediction-zone">
                <div class="result-label">Predicted Category</div>
                <div class="main-prediction-pill" style="background:{pill_color};">
                    {html_escape.escape(predicted_display)}
                </div>
            </div>

            <div class="confidence-ring">
                <div class="confidence-number">{confidence_display}</div>
                <div class="confidence-text">{confidence_subtitle}</div>
            </div>
        </div>

        <div class="status-strip">
          <div class="status-chip completed">Prediction Completed</div>
          <div class="status-chip">Language Detected</div>
          <div class="status-chip">Model Selected</div>
        </div>

        <div class="result-metadata-grid three-grid">
            <div class="metadata-tile">
                <div class="result-label">Detected Language</div>
                <div class="result-value compact">{html_escape.escape(result.detected_language)}</div>
            </div>

            <div class="metadata-tile">
                <div class="result-label">Selected Model</div>
                <div class="result-value compact">{html_escape.escape(selected_model_display)}</div>
            </div>

            <div class="metadata-tile">
                <div class="result-label">Decision Strength</div>
                <div class="strength-pill {model_strength_class}">
                    {html_escape.escape(model_strength_display)}
                </div>
            </div>
        </div>
    </div>
    """)


def render_research_footer():
    html("""
    <div class="research-footer">
        <div class="footer-title">Research Use Notice</div>
        <div class="footer-text">
            This web interface is designed for final-year project demonstration and model evaluation.
            It does not provide medical diagnosis, clinical judgment, or emergency support.
        </div>
    </div>
    """)


load_css()

if "last_prediction_response" not in st.session_state:
    st.session_state.last_prediction_response = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "last_text" not in st.session_state:
    st.session_state.last_text = ""


html("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">FYP Research Interface</div>
        <div class="hero-title">Cross-Lingual Mental Health Classifier</div>
        <div class="hero-subtitle">
            A research web interface for testing whether English, French, or Arabic written text
            is associated with a mental health related category.
        </div>

        <div class="language-row">
            <div class="language-chip chip-en">English</div>
            <div class="language-chip chip-fr">French</div>
            <div class="language-chip chip-ar">Arabic</div>
        </div>
    </div>
</div>
""")

left_col, right_col = st.columns([0.92, 1.18])

with left_col:
    html("""
    <div class="section-card input-info-card">
        <div class="eyebrow">Input Layer</div>
        <div class="section-title">Input Analysis</div>
        <div class="section-text">
            Enter text in English, French, or Arabic. The system automatically detects
            the language and routes the input to the corresponding classifier.
        </div>
    </div>
    """)

    if DEMO_MODE:
        html("""
        <div class="colored-note">
            Demo mode is currently active. The interface is ready, but real model files are not connected yet.
        </div>
        """)

    text_input = st.text_area(
        label="Input to analyze",
        height=155,
        placeholder="Paste text in Arabic, English, or French..."
    )

    analyze_clicked = st.button("Analyze Text")

    if analyze_clicked:
        if not text_input.strip():
            st.session_state.last_error = "Please enter text before analyzing."
            st.session_state.last_prediction_response = None
            st.session_state.last_text = ""
        else:
            prediction_response = predict_text(text_input.strip())
            st.session_state.last_prediction_response = prediction_response
            st.session_state.last_error = None
            st.session_state.last_text = text_input.strip()

    current_result = None
    if st.session_state.last_prediction_response and st.session_state.last_prediction_response.get("ok"):
        current_result = st.session_state.last_prediction_response["result"]

    render_pipeline_card(current_result)

with right_col:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
        render_empty_result()
    elif st.session_state.last_prediction_response:
        prediction_response = st.session_state.last_prediction_response

        if prediction_response["ok"]:
            result = prediction_response["result"]
            render_result(result)
            render_probability_card(result)
        else:
            st.error(prediction_response["error"])
            render_empty_result()
    else:
        render_empty_result()

render_research_footer()