import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import os

def generate_test_data(num_samples=5000):
    """Generates fresh, unseen data to test the model."""
    np.random.seed(99) # Different seed than training
    hour_of_day = np.random.randint(0, 24, num_samples)
    day_of_week = np.random.randint(0, 7, num_samples)
    is_raining = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    base_price = np.random.uniform(5.0, 25.0, num_samples)

    availability_prob = 0.85 
    availability_prob -= np.where((hour_of_day >= 8) & (hour_of_day <= 18), 0.35, 0.0)
    availability_prob -= np.where(day_of_week >= 5, 0.20, 0.0)
    availability_prob -= (is_raining * 0.15)
    availability_prob += (base_price / 100)
    availability_prob = np.clip(availability_prob, 0.05, 0.95)

    is_available = np.random.binomial(1, availability_prob)

    return pd.DataFrame({
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_raining': is_raining,
        'base_price': base_price,
        'is_available': is_available
    })

def evaluate_xgboost():
    # 1. Load the Model
    model_path = "../data/parking_model.json"
    if not os.path.exists(model_path):
        model_path = "data/parking_model.json"
        
    if not os.path.exists(model_path):
        print("Model not found! Please run train_parking_model.py first.")
        return

    print("Loading XGBoost Model...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # 2. Get Unseen Test Data
    print("Generating unseen test data...")
    df_test = generate_test_data()
    X_test = df_test[['hour_of_day', 'day_of_week', 'is_raining', 'base_price']]
    y_test = df_test['is_available']

    # 3. Make Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (Available)

    # --- PLOTTING ---
    plt.figure(figsize=(18, 5))
    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Full (0)', 'Available (1)'],
                yticklabels=['Full (0)', 'Available (1)'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Plot 2: ROC Curve (Receiver Operating Characteristic)
    plt.subplot(1, 3, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")

    # Plot 3: Feature Importance
    plt.subplot(1, 3, 3)
    feature_importances = model.feature_importances_
    features = X_test.columns
    indices = np.argsort(feature_importances)
    
    plt.barh(range(len(indices)), feature_importances[indices], color='mediumseagreen')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance')

    plt.tight_layout()
    
    # Save the plot for your report and show it
    save_path = "data/model_evaluation_metrics.png" if os.path.exists("data") else "../data/model_evaluation_metrics.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Evaluation complete! Graphs saved to: {save_path}")
    
    # Print the raw classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    plt.show()

if __name__ == "__main__":
    evaluate_xgboost()