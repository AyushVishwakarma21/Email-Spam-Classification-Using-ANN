import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from src.preprocessing import clean_text, pad_text_sequences, load_tokenizer
from src.feedback_utils import save_feedback_csv

MODEL_PATH = 'models/spam_cnn_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'
MAX_LEN = 200

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    return model, tokenizer

model, tokenizer = load_artifacts()

def classify_email(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_text_sequences(seq, max_len=MAX_LEN)
    prob = model.predict(pad)[0][0]
    label = "SPAM" if prob > 0.5 else "HAM"
    return label, prob

st.title("📧 Email Spam Classifier (CNN) with Optional Feedback")

email_input = st.text_area("✍️ Enter email content here:", height=200)

if st.button("🚀 Classify"):
    if email_input.strip() == "":
        st.warning("Please enter some email text first.")
    else:
        prediction, confidence = classify_email(email_input)
        st.success(f"Prediction: **{prediction}** (Confidence: {confidence:.2f})")

        with st.expander("📝 Optional: Was this prediction correct?"):
            feedback = st.radio("Select one if you want to submit feedback:", ('', 'Yes', 'No'), index=0)

            if feedback in ['Yes', 'No']:
                if st.button("💾 Submit Feedback"):
                    save_feedback_csv(email_input, prediction, feedback)
                    st.info("Thank you for your feedback! It has been saved.")
