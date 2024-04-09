from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load the trained model
model = tf.keras.models.load_model('D:/settel.h5')

# Load Tokenizer if needed
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["your list of external status descriptions"])

# Define request body model
class Item(BaseModel):
    description: str

# 3. Implement API endpoint
@app.post("/predict")
async def predict(item: Item):
    # 4. Preprocess the input data
    text = [item.description]
    sequence = tokenizer.texts_to_sequences(text)
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

    # 5. Use the loaded model to make predictions
    prediction = model.predict(padded_sequence)

    # Assuming prediction is binary, you can adjust this part accordingly
    predicted_label = "internal_status_label_1" if prediction[0][0] >= 0.5 else "internal_status_label_2"

    return {"predicted_internal_status": predicted_label}
# 6. Run the API with uvicorn 



# uvicorn main:app --reload
