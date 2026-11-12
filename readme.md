# Claim Outcome Prediction Pipeline

This repository contains an end-to-end machine learning pipeline that simulates historical healthcare claims, trains a classification model to predict claim outcomes, and serves the trained model via a FastAPI endpoint.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

2. Generate data and train the model:
This will simulate the dataset, engineer features (preventing data leakage), train the RandomForestClassifier, and save the model_artifacts.joblib file.   
    python train.py
3. Start the API server:
   uvicorn app:app --reload
4. Test the Endpoint:
You can test the API using Postman, cURL, or the built-in FastAPI docs at http://127.0.0.1:8000/docs. Send a POST request to /predict with the following JSON body:
{
  "payer": "Aetna",
  "provider": "Dr.Rao",
  "cpt_code": "99213",
  "diagnosis_code": "M54.5",
  "billed_amount": 220.00,
  "patient_age": 45
}


## Evaluation Results
Because the dataset is intentionally imbalanced (70% Paid, 20% Partially_Paid, 10% Denied), the model's performance was evaluated using precision, recall, and F1-score rather than plain accuracy.
precision    recall  f1-score   support
        Denied       0.00      0.00      0.00         9
          Paid       0.69      0.90      0.78        68
Partially_Paid       0.40      0.17      0.24        23

      accuracy                           0.65       100

Confusion Matrix:
[[ 0  8  1] 
 [ 2 61  5] 
 [ 0 19  4]]

 ## Model Weakness & Improvements
     The model is weakest at predicting the Denied class, achieving a recall of exactly 0.00. Because the dataset is highly imbalanced, the model heavily biased its predictions toward the majority "Paid" class to maximize accuracy, completely missing actual denials. To improve this, I would apply SMOTE (Synthetic Minority Over-sampling Technique) to synthetically oversample the Denied claims in the training data, or lower the classification probability threshold specifically for the Denied class to make the model more sensitive to it.    
## Shortcuts & Assumptions
 ## Simulated Data Noise: 
      Because I generated the dataset using purely random distributions without underlying mathematical correlations, the ML model is essentially learning from noise. The poor metrics are a reflection of this random generation rather than a flaw in the pipeline architecture.   
## Error Handling: 
     I assumed that automatically returning a 422 Unprocessable Entity via FastAPI's Pydantic validation for missing or incorrectly typed fields safely satisfies the requirement to return a "clear 400-style error".  
## Feature Engineering: 
     Features like payer_denial_rate were calculated strictly on the training set and mapped to the test set to avoid data leakage.
