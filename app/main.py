import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()
model = joblib.load("models/models_churn_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "probability": round(float(probability), 2)
    }
