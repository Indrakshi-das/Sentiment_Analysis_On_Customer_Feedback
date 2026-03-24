import re
import unicodedata
import joblib
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42

st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl_1")
    vectorizer = joblib.load("tfidf_vectorizer.pkl_1")
    return model, vectorizer

model, vectorizer = load_artifacts()


# -----------------------------
# Text cleaning + encoding repair
# -----------------------------
def fix_mojibake(text: str) -> str:
    """
    Repair common UTF-8 text that was wrongly decoded as latin-1/cp1252.
    Works for many broken Chinese/Malay/other Unicode strings.
    """
    if text is None:
        return ""

    text = str(text)

    suspicious_markers = ["Ã", "Â", "â", "è", "é", "ç", "å", "¤", "¼", "™", "ð", "Ÿ"]
    if not any(marker in text for marker in suspicious_markers):
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        pass

    try:
        return text.encode("cp1252").decode("utf-8")
    except Exception:
        pass

    return text


def normalize_text(x: str) -> str:
    if x is None:
        return ""

    x = str(x)
    x = fix_mojibake(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.replace("\u00a0", " ")
    x = x.replace("Â", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


# -----------------------------
# Language detection + translation
# -----------------------------
def safe_detect_language(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "unknown"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def safe_translate_to_english(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text

    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def language_name(code: str) -> str:
    mapping = {
        "en": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "ms": "Malay",
        "id": "Indonesian",
        "ta": "Tamil",
        "hi": "Hindi",
        "fi": "Unknown / Misdetected",
        "unknown": "Unknown"
    }
    return mapping.get(code, code)


# -----------------------------
# Question grouping
# -----------------------------
def classify_question_group(question: str) -> str:
    q = normalize_text(question).lower()

    if "recommend" in q or "likely" in q:
        return "rating_intent"
    elif "satisfied" in q or "easy" in q or "agree" in q:
        return "structured_sentiment"
    elif "improve" in q or "comment" in q or "suggest" in q:
        return "open_feedback"
    else:
        return "other"


# -----------------------------
# Prediction
# -----------------------------
def predict_sentiment(question: str, answer: str):
    question_clean = normalize_text(question)
    answer_clean = normalize_text(answer)

    q_lang_code = safe_detect_language(question_clean)
    a_lang_code = safe_detect_language(answer_clean)

    question_en = question_clean if q_lang_code == "en" else safe_translate_to_english(question_clean)
    answer_en = answer_clean if a_lang_code == "en" else safe_translate_to_english(answer_clean)

    question_group = classify_question_group(question_en)

    lang_feature = "english" if (q_lang_code == "en" and a_lang_code == "en") else "translated"

    final_input = (
        "[LANG] " + lang_feature +
        " [QGROUP] " + question_group +
        " [QUESTION] " + question_en +
        " [ANSWER] " + answer_en
    )

    vec = vectorizer.transform([final_input])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    inv_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return {
        "question_clean": question_clean,
        "answer_clean": answer_clean,
        "question_lang": language_name(q_lang_code),
        "answer_lang": language_name(a_lang_code),
        "translated_question": question_en,
        "translated_answer": answer_en,
        "question_group": question_group,
        "final_input": final_input,
        "prediction": inv_label_map[pred],
        "probabilities": {
            "Negative": float(probs[0]),
            "Neutral": float(probs[1]),
            "Positive": float(probs[2]),
        }
    }


# -----------------------------
# UI
# -----------------------------
st.title("💬 Multilingual Sentiment Analysis")
st.write(
    "Enter any question and answer. If the input is in Chinese, Malay, or another language, "
    "the app will first detect the language, translate it to English, and then predict sentiment."
)

question = st.text_area(
    "Question",
    placeholder="How can I help you?"
)

answer = st.text_area(
    "Answer",
    placeholder="Ask an answer"
)

predict_btn = st.button("Predict Sentiment")

if predict_btn:
    if not question.strip() or not answer.strip():
        st.warning("Please enter both question and answer.")
    else:
        with st.spinner("Cleaning, translating, and predicting..."):
            result = predict_sentiment(question, answer)

        st.subheader("Result")
        st.success(f"Predicted Sentiment: **{result['prediction']}**")

        st.subheader("Language Detection")
        st.write(f"Question Language: **{result['question_lang']}**")
        st.write(f"Answer Language: **{result['answer_lang']}**")

        st.subheader("Cleaned / Repaired Text")
        st.write("**Cleaned Question:**")
        st.write(result["question_clean"])
        st.write("**Cleaned Answer:**")
        st.write(result["answer_clean"])

        st.subheader("Translated Text")
        st.write("**Translated Question:**")
        st.write(result["translated_question"])
        st.write("**Translated Answer:**")
        st.write(result["translated_answer"])

        st.subheader("Model Input Summary")
        st.write(f"**Question Group:** {result['question_group']}")

        with st.expander("Show final model input"):
            st.code(result["final_input"])

        st.subheader("Confidence Scores")
        st.write(f"Negative: **{result['probabilities']['Negative']:.4f}**")
        st.write(f"Neutral: **{result['probabilities']['Neutral']:.4f}**")
        st.write(f"Positive: **{result['probabilities']['Positive']:.4f}**")