import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Generate Dataset
np.random.seed(42)
n_rows = 500
data = {
    'payer': np.random.choice(['Aetna', 'Cigna', 'UnitedHC', 'BlueCross'], n_rows),
    'provider': np.random.choice(['Dr.Rao', 'Dr.Shah', 'Dr.Lee', 'Dr.Kim'], n_rows),
    'cpt_code': np.random.choice(['99213', '99214', '99215', '99203'], n_rows),
    'diagnosis_code': np.random.choice(['M54.5', 'J06.9', 'E11.9', 'I10'], n_rows),
    'billed_amount': np.round(np.random.uniform(100, 500, n_rows), 2),
    'patient_age': np.random.randint(18, 80, n_rows),
    'status': np.random.choice(['Paid', 'Partially_Paid', 'Denied'], n_rows, p=[0.70, 0.20, 0.10])
}
df = pd.DataFrame(data)

# 2. Prepare Data
X = df.drop(columns=['status'])
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Engineered Features
train_df = X_train.copy()
train_df['status'] = y_train
payer_denial_rate = (train_df[train_df['status'] == 'Denied'].groupby('payer').size() / train_df.groupby('payer').size()).fillna(0).to_dict()
provider_avg_billed = train_df.groupby('provider')['billed_amount'].mean().to_dict()

def apply_features(X_df):
    df_feat = X_df.copy()
    df_feat['payer_denial_rate'] = df_feat['payer'].map(payer_denial_rate).fillna(0)
    df_feat['provider_avg_billed'] = df_feat['provider'].map(provider_avg_billed).fillna(df_feat['billed_amount'].mean())
    return df_feat

X_train_eng = apply_features(X_train)

# 3. Train and Save Model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['billed_amount', 'patient_age', 'payer_denial_rate', 'provider_avg_billed']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['payer', 'provider', 'cpt_code', 'diagnosis_code'])
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

pipeline.fit(X_train_eng, y_train)

joblib.dump({
    'pipeline': pipeline,
    'payer_denial_rate': payer_denial_rate,
    'provider_avg_billed': provider_avg_billed,
    'classes': pipeline.classes_.tolist()
}, 'model_artifacts.joblib')

print("Model successfully retrained and saved")