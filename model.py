import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from label_sentiment import label_sentiment
from predict_sentiment import predict_sentiment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


data = pd.read_csv('dataset/1429_1.csv')
dataset = data[["id","reviews.text","reviews.rating"]]

dataset = dataset.dropna()
dataset['label'] = dataset['reviews.rating'].map(label_sentiment)

text = " ".join(dataset['reviews.text']).lower().split('.')

# Split data
X = dataset['reviews.text'].astype(str).values
y = dataset['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(y)

# Tokenize (Building the dictionary)
max_words = 13000
tokenizer = Tokenizer(num_words = max_words, oov_token = "<OOV>")
tokenizer.fit_on_texts(X_train)

total_unique_words = len(tokenizer.word_index)
print(f"Total unique words in my data: {total_unique_words}")

# Sequence and Pad
maxlen = 500
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = maxlen)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = maxlen)  

embedding_dim = 128
model = Sequential([
    # Layer 1: Word Embeddings (Learns word meanings from scratch)
    Embedding(max_words, embedding_dim, input_length = maxlen),
    SpatialDropout1D(0.2),
    
    # Layer 2: LSTM (Remembers context across the review)
    LSTM(128, dropout = 0.2),
    
    # Layer 3
    Dense(64, activation = 'relu'),
    Dropout(0.2),
    
    # Layer 4: Final Output (3 classes)
    Dense(3, activation = 'softmax')
])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics =['accuracy']
)


model.build(input_shape=(None, maxlen)) 
model.summary()

y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)


early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    X_train_seq, y_train_encoded,
    epochs = 1,
    batch_size = 32,
    validation_data = (X_test_seq, y_test_encoded),
    callbacks = [early_stop]
)

# Save model
model.save("sentiment_model.h5")