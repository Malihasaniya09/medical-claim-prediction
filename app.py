from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model without retraining from scratch[cite: 2].
try:
    artifacts = joblib.load('model_artifacts.joblib')
    pipeline = artifacts['pipeline']
    payer_denial_rate = artifacts['payer_denial_rate']
    provider_avg_billed = artifacts['provider_avg_billed']
    classes = artifacts['classes']
except FileNotFoundError:
    pipeline = None

# Pydantic enforces types. Missing/invalid fields automatically return a clear 400-level error (422 Unprocessable Entity) without crashing.
class Claim(BaseModel):
    payer: str
    provider: str
    cpt_code: str
    diagnosis_code: str
    billed_amount: float
    patient_age: int

@app.post("/predict")
def predict(claim: Claim):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Model artifacts not found.")

    input_data = pd.DataFrame([claim.model_dump()])

    input_data['payer_denial_rate'] = input_data['payer'].map(payer_denial_rate).fillna(0)
    input_data['provider_avg_billed'] = input_data['provider'].map(provider_avg_billed).fillna(input_data['billed_amount'].mean())

    probs = pipeline.predict_proba(input_data)[0]
    pred_idx = probs.argmax()
    predicted_status = classes[pred_idx]

    prob_dict = {classes[i]: round(float(probs[i]), 2) for i in range(len(classes))}

    # Returns the predicted status alongside the probabilities for each class.
    return {
        "predicted_status": predicted_status,
        "probabilities": prob_dict
    }