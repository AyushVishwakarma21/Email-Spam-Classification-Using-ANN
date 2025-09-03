import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
import os
from src.evaluate import evaluate_model


# Import from src
from src.preprocessing import clean_text, tokenize_texts, pad_text_sequences, save_tokenizer
from src.model import build_cnn_model

# Create model directory
os.makedirs("models", exist_ok=True)

# 1. Load data
df = pd.read_csv('data/emails.csv')
df['clean_text'] = df['text'].apply(clean_text)
labels = df['label'].values

# 2. Tokenize and pad
VOCAB_SIZE = 10000
MAX_LEN = 200

tokenizer, sequences = tokenize_texts(df['clean_text'], vocab_size=VOCAB_SIZE)
padded = pad_text_sequences(sequences, max_len=MAX_LEN)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, stratify=labels, random_state=42)

# 4. Train model
model = build_cnn_model(vocab_size=VOCAB_SIZE, embedding_dim=100, max_len=MAX_LEN)
model.fit(X_train, y_train, epochs=5, batch_size=512, validation_split=0.2)

# 5. Save model/tokenizer
save_model(model, 'models/spam_cnn_model.h5')
save_tokenizer(tokenizer, 'models/tokenizer.pkl')


# Evaluate and save confusion matrix
evaluate_model(model, X_test, y_test)
