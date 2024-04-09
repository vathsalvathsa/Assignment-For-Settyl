import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Data Preparation
df = pd.read_csv('C:/Users/Sundram Vaths/Desktop/Assignment For Settyl/updated_csv_file.csv')

# Assuming 'externalStatus' is the target label and 'description' is the input data
X = df['internalStatus']
y = df['externalStatus']

# 2. Data Preprocessing
vocab_size = 10000  # Adjust as per vocabulary size
max_length = 100    # Adjust as per the maximum length of sequences
embedding_dim = 16  # Adjust as per the embedding dimension

X = X.astype(str)
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OO>')
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

# 3. Model Building
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 4. Model Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

# 5. Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# 6. Model Deployment
model.save('D:/settel.h5') 

