# End to End Deep Learning Project Using Simple RNN 

# Step 1: Importing Libraries 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Step 2: Loading and Preprocessing the IMDB dataset
max_features = 10000 # Vocabulary size
max_len = 500 # Number of words in sentences
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

## Inspect a sample review and its label
sample_review=X_train[0]
sample_label=y_train[0]
# MApping of words index bacl to words(for understanding)
word_index=imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])

X_train=sequence.pad_sequences(X_train,maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# Step 3: Training the Simple RNN Model
model=Sequential()
model.add(Embedding(max_features,128,input_length=max_len)) ## Embedding Layers
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Create an instance of EarlyStoppping Callback
from tensorflow.keras.callbacks import EarlyStopping
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# Train the model with early stopping
history=model.fit(
    X_train,y_train,epochs=10,batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)

# Step 4: Save model file
model.save('simple_rnn_imdb.h5')