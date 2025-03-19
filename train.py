import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\Admin\Downloads\ghatkopar_events_dataset.csv")

# Convert Month names to numeric values
df['Month'] = df['Month'].apply(lambda x: pd.to_datetime(x, format='%B').month)

# Convert Year-Month to Date format
df['ds'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
df = df[['ds', 'Event Type', 'Event Count']]

# Train models for each event type
models = {}

for event in df['Event Type'].unique():
    event_df = df[df['Event Type'] == event].copy()
    
    # Prophet Model
    prophet_model = Prophet()
    prophet_model.fit(event_df[['ds', 'Event Count']].rename(columns={'Event Count': 'y'}))
    
    # XGBoost Model
    event_df['month'] = event_df['ds'].dt.month
    event_df['year'] = event_df['ds'].dt.year
    
    X = event_df[['month', 'year']]
    y = event_df['Event Count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    
    models[event] = {'prophet': prophet_model, 'xgb': xgb_model}

# Save models
with open("models/event_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("✅ AI Models Trained & Saved!")
