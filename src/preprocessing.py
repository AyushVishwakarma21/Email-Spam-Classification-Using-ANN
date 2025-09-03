import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

def clean_text(text):
    """
    Basic text cleaning: lowercasing, removing HTML tags, URLs, extra spaces.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_texts(texts, vocab_size=10000, oov_token='<OOV>'):
    """
    Fit a tokenizer on texts and return tokenizer + sequences.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return tokenizer, sequences

def pad_text_sequences(sequences, max_len=200):
    """
    Pad/truncate sequences to a fixed max length.
    """
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

def save_tokenizer(tokenizer, path='models/tokenizer.pkl'):
    """
    Save the tokenizer object to file.
    """
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path='models/tokenizer.pkl'):
    """
    Load a tokenizer object from file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
