import streamlit as st
import os
import pickle
import pandas as pd
import time
import streamlit.components.v1 as components

# Load the model and threshold
@st.cache_resource
def load_model():
    file_path = os.path.join(os.path.dirname(__file__), 'voting_pipeline_with_threshold.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)

saved = load_model()
voting_pipeline = saved["pipeline"]
threshold = saved["threshold"]

def predict_with_threshold(pipeline, X, threshold = 0.80):
    proba = pipeline.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)

# Streamlit App
st.title("Email Spam Classifier")

input_email =  st.text_area("Enter your email")

if st.button("Predict"):
    if input_email.strip():
        input_df = pd.DataFrame({'email_text': [input_email.strip()]})
        prediction = predict_with_threshold(voting_pipeline, input_df, threshold)[0]
        if prediction == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

        # Wait and clear
        time.sleep(5)
        components.html(
            """<script>
            const textarea = parent.document.querySelector('textarea');
            if (textarea) textarea.value = "";
            </script>""",
            height = 0,
        )
    else:
        st.warning("Please enter a message!")
