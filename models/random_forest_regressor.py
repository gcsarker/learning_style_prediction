import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib

def train(training_data, validation_data=None, random_state=2025):
    # training data is a dictionary
    X_train = training_data['X_train']  # NumPy array
    y_rgr_train = training_data['y_rgr_train']  # NumPy array

    checkpoint_path = "checkpoints"

    # Initialize RandomForestRegressor for each output (4 outputs)
    print("Loading Random Forest Regressor")
    rf_models = []
    for i in range(y_rgr_train.shape[1]):  # Iterate through each output column
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)  # You can adjust parameters
        rf.fit(X_train, y_rgr_train[:, i])  # Train on one output at a time (NumPy indexing)
        rf_models.append(rf)

    # Save the models
    joblib.dump(rf_models, os.path.join(checkpoint_path, "rf_regressors.pkl"))

    print('Training Successful!')

    return True

def evaluate(test_data, scaler_X, scaler_y, class_names, checkpoint_path='checkpoints'):
    X_test = test_data['X_test']
    y_test_rgr = test_data['y_rgr_test']

    # Standard scaling input X
    X_test_scaled = scaler_X.transform(X_test.values) 

    # Load the models
    rf_models = joblib.load(os.path.join(checkpoint_path, "rf_regressors.pkl"))

    # Make predictions for each output
    y_pred_rgr_scaled = np.zeros((X_test_scaled.shape[0], len(rf_models)))  # Initialize prediction array
    for i, rf in enumerate(rf_models):
        y_pred_rgr_scaled[:, i] = rf.predict(X_test_scaled)

    y_pred_rgr = scaler_y.inverse_transform(y_pred_rgr_scaled)

    # Evaluation for regression Task
    print('Regression Evaluation\n')
    for i, target in enumerate(class_names):
        mse = mean_squared_error(y_test_rgr.iloc[:, i], y_pred_rgr[:, i])
        r2 = r2_score(y_test_rgr.iloc[:, i], y_pred_rgr[:, i])
        mae = mean_absolute_error(y_test_rgr.iloc[:,i], y_pred_rgr[:, i])
        print(f"\nTarget: {target}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f'Mean Absolute Error: {mae:.4f}')

    return True