import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -- Load Assets --
@st.cache_resource
def load_assets():
    model = load_model('sentiment_model_new.h5')
    with open('tokenizer_new.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

model, tokenizer = load_assets()

# -- App UI --
st.title("😊 Sentiment Predictor 😠")
st.write("Enter a review to see if it is Positive or Negative.")

user_input = st.text_area("Your review:", placeholder="Type here...")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        # 1. Preprocess
        maxlen = 500
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=maxlen)
        
        # 2. Predict
        prediction = model.predict(padded)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        
        # 3. Display Results with Alignment
        st.write("") # Small spacer
        
        if class_idx == 1:
            # Combined text inside the success box for perfect alignment
            st.success(f"### Positive Sentiment 🟢  \n**Confidence:** {confidence:.2f}%")
        else:
            # Combined text inside the error box for perfect alignment
            st.error(f"### Negative Sentiment 🔴  \n**Confidence:** {confidence:.2f}%")
        
    else:
        st.warning("Please enter some text first!")