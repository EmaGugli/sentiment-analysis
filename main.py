from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
#import numpy as np

app = FastAPI()

embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
label_encoder = joblib.load("models/label_encoder.joblib")
model = joblib.load("models/logistic_regression_model.joblib")

class Item(BaseModel):
    text: str

@app.post("/sentiment_analysis")
def predict(item: Item):
    vector = embedder.encode(item.text).reshape(1, -1)
    prediction = model.predict(vector)
    label = label_encoder.inverse_transform(prediction)
    return {"prediction": label[0]}
