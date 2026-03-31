import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

print("Training ETA Confidence Models natively on Windows...")

# 1. Generate the same training data we used in Colab
num_samples = 5000
np.random.seed(42)

X_eta = pd.DataFrame({
    'Distance_km': np.random.uniform(1.0, 15.0, num_samples),
    'Traffic_Risk': np.random.uniform(0.0, 1.0, num_samples),
    'Hour': np.random.randint(0, 24, num_samples),
    'Is_Raining': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
})

base_time = X_eta['Distance_km'] * 3.0
traffic_delay = X_eta['Traffic_Risk'] * 15.0
weather_penalty = 1.0 + (X_eta['Is_Raining'] * 0.2)

y_eta = (base_time + traffic_delay) * weather_penalty + np.random.normal(0, 3.0, num_samples)
y_eta = np.maximum(y_eta, 1.0) 

# 2. Train the models
gbm_median = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100).fit(X_eta, y_eta)
gbm_lower = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100).fit(X_eta, y_eta)
gbm_upper = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100).fit(X_eta, y_eta)

# 3. Save them directly into the data folder
os.makedirs("data", exist_ok=True)
joblib.dump(gbm_median, 'data/gbm_eta_median.pkl')
joblib.dump(gbm_lower, 'data/gbm_eta_lower.pkl')
joblib.dump(gbm_upper, 'data/gbm_eta_upper.pkl')

print("✅ Native .pkl files successfully generated in your data/ folder!")