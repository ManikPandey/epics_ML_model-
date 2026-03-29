import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def generate_synthetic_parking_data(num_samples=15000):
    """
    Generates realistic historical parking data.
    In the real world, this data comes from City IoT sensors or TLC Trip records.
    """
    print(f"Generating {num_samples} rows of historical parking data...")
    np.random.seed(42)

    # Features: The context of the city
    hour_of_day = np.random.randint(0, 24, num_samples)
    day_of_week = np.random.randint(0, 7, num_samples) # 0=Monday, 6=Sunday
    is_raining = np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 20% chance of rain
    base_price = np.random.uniform(5.0, 25.0, num_samples)

    # Target Variable Logic: 
    # Harder to park during midday (8-18), harder on weekends, harder when raining
    availability_prob = 0.85 # Base probability it is empty
    
    # Apply real-world penalties
    availability_prob -= np.where((hour_of_day >= 8) & (hour_of_day <= 18), 0.35, 0.0) # Rush hour penalty
    availability_prob -= np.where(day_of_week >= 5, 0.20, 0.0) # Weekend penalty
    availability_prob -= (is_raining * 0.15) # Rain means people drive instead of walk
    availability_prob += (base_price / 100) # Expensive spots are slightly more likely to be empty

    # Cap probabilities between 5% and 95%
    availability_prob = np.clip(availability_prob, 0.05, 0.95)

    # Generate the actual binary outcome (1 = Available, 0 = Taken) based on the calculated probability
    is_available = np.random.binomial(1, availability_prob)

    df = pd.DataFrame({
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_raining': is_raining,
        'base_price': base_price,
        'is_available': is_available
    })
    return df

def train_and_save_model():
    """Trains the XGBoost Classifier and saves the weights to disk."""
    df = generate_synthetic_parking_data()

    # Define Features (X) and Target (y)
    X = df[['hour_of_day', 'day_of_week', 'is_raining', 'base_price']]
    y = df['is_available']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost Spatiotemporal Predictor...")
    # Initialize the model
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model Training Complete! Accuracy: {acc * 100:.2f}%")

    # Save the model to our data folder so FastAPI can load it
    # Handle path routing depending on where the user runs the script
    model_path = "../data/parking_model.json"
    if not os.path.exists("../data"):
        os.makedirs("data", exist_ok=True)
        model_path = "data/parking_model.json"
        
    model.save_model(model_path)
    print(f"💾 Model weights successfully saved to: {model_path}")

if __name__ == "__main__":
    train_and_save_model()