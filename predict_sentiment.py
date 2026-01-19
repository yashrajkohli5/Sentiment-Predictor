import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(text):
   seq = tokenizer.texts_to_sequences([text])
   padded = pad_sequences(seq, maxlen = maxlen)
   prediction = model.predict(padded, verbose = 0)
   
   class_idx = np.argmax(prediction, axis = 1)[0]
   
   mapping = {0: "Negative", 1: "Positive"}
   return mapping[class_idx] 
