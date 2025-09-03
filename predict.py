from tensorflow.keras.models import load_model
from src.preprocessing import clean_text, pad_text_sequences, load_tokenizer

MAX_LEN = 200

# Load model and tokenizer
model = load_model('models/spam_cnn_model.h5')
tokenizer = load_tokenizer('models/tokenizer.pkl')

def predict_email(text):
    """
    Predict SPAM or HAM for a given email text.
    """
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_text_sequences(seq, max_len=MAX_LEN)
    prob = model.predict(pad)[0][0]
    return "SPAM" if prob > 0.5 else "HAM", float(prob)

if __name__ == "__main__":
    email_text = input("Enter an email to classify:\n")
    label, confidence = predict_email(email_text)
    print(f"\nPrediction: {label} (Confidence: {confidence:.2f})")
