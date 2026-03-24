import re
import unicodedata
import joblib
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42


st.set_page_config(page_title="Multilingual Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()


def normalize_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.replace("\u00a0", " ")
    x = x.replace("Â", " ")  
    x = re.sub(r"\s+", " ", x).strip()
    return x

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

def language_name(code: str) -> str:
    mapping = {
        "en": "English",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "zh": "Chinese",
        "ms": "Malay",
        "id": "Indonesian",
        "ta": "Tamil",
        "hi": "Hindi",
        "unknown": "Unknown"
    }
    return mapping.get(code, code)

def predict_sentiment(question: str, answer: str):
 
    question_clean = normalize_text(question)
    answer_clean = normalize_text(answer)

    q_lang = safe_detect_language(question_clean)
    a_lang = safe_detect_language(answer_clean)


    question_en = question_clean if q_lang == "en" else safe_translate_to_english(question_clean)
    answer_en = answer_clean if a_lang == "en" else safe_translate_to_english(answer_clean)

    question_group = classify_question_group(question_en)
    lang_feature = "english" if (q_lang == "en" and a_lang == "en") else "translated"

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
        "question_lang": language_name(q_lang),
        "answer_lang": language_name(a_lang),
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

st.title("💬 Multilingual Sentiment Analysis")
st.write("Enter a question and answer. If the text is in Chinese, Malay, or another language, it will be translated to English before prediction.")

question = st.text_area(
    "Question",
    placeholder="Example: What can Mah Sing do to improve your satisfaction with our Customer Service Representatives?"
)

answer = st.text_area(
    "Answer",
    placeholder="Example: 聘请多点各种族的人 了解沟通比较容易"
)

predict_btn = st.button("Predict Sentiment")

if predict_btn:
    if not question.strip() or not answer.strip():
        st.warning("Please enter both question and answer.")
    else:
        with st.spinner("Translating and predicting..."):
            result = predict_sentiment(question, answer)

        st.subheader("Result")
        st.success(f"Predicted Sentiment: **{result['prediction']}**")

        st.subheader("Language Detection")
        st.write(f"Question Language: **{result['question_lang']}**")
        st.write(f"Answer Language: **{result['answer_lang']}**")

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