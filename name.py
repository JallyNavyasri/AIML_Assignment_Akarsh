import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier # CHANGE 1
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("data/daily_targeting.csv")

status_col = [col for col in df.columns if 'status' in col.lower()][0]
df['Default'] = np.where(df[status_col].isin(['EXPIRED', 'TERMINATED']), 1, 0)

feature_cols = ['priority']
if 'campaign_id' in df.columns:
    df['campaign_id'] = pd.factorize(df['campaign_id'])[0]
    feature_cols.append('campaign_id')
if 'target_date' in df.columns:
    df['target_date'] = pd.to_datetime(df['target_date'], errors='coerce').dt.dayofweek.fillna(0)
    feature_cols.append('target_date')

X = df[feature_cols].fillna(0)
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# CHANGE 2: XGBoost Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy = {round(acc*100, 2)}%")
print(classification_report(y_test, y_pred))

os.makedirs("notebooks", exist_ok=True)
plt.figure(figsize=(8,6))
sns.heatmap(df[feature_cols + ['Default']].corr(), annot=True, cmap="coolwarm")
plt.savefig("notebooks/correlation.png")

joblib.dump(model, "notebooks/best_model.pkl")
print("Graph + Model Saved ✅")