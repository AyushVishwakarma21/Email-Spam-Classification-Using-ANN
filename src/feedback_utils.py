import os
import pandas as pd

FEEDBACK_PATH = 'outputs/feedback.csv'

def save_feedback_csv(email_text, prediction, correct):
    os.makedirs('outputs', exist_ok=True)

    feedback_data = {
        'email_text': [email_text],
        'prediction': [prediction],
        'user_feedback_correct': [correct]
    }

    df_new = pd.DataFrame(feedback_data)

    if os.path.exists(FEEDBACK_PATH):
        df_existing = pd.read_csv(FEEDBACK_PATH)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(FEEDBACK_PATH, index=False)
