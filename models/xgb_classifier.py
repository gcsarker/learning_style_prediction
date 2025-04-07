import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, precision_score, recall_score, f1_score
import joblib

def train(training_data, validation_data=None, random_state=2025):
    X_train = training_data['X_train']
    y_cls_train = training_data['y_cls_train']

    checkpoint_path = "checkpoints"

    # Train a separate XGBClassifier for each label
    print("Loading XGBoost Multilabel Classifiers")
    classifiers = []
    for i in range(y_cls_train.shape[1]):
        classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=random_state)
        classifier.fit(X_train, y_cls_train[:, i])
        classifiers.append(classifier)

    # Save the models
    joblib.dump(classifiers, os.path.join(checkpoint_path, "xgb_multilabel_cls.pkl"))

    print('Training Successful!')

    return True

def evaluate(test_data, scaler_X, scaler_y, class_names, checkpoint_path='checkpoints'):
    X_test = test_data['X_test']
    y_test_cls = test_data['y_cls_test']

    # Standard scaling input X
    X_test_scaled = scaler_X.transform(X_test.values)

    # Load the models
    classifiers = joblib.load(os.path.join(checkpoint_path, "xgb_multilabel_cls.pkl"))

    # Make predictions for each label
    y_pred = np.zeros_like(y_test_cls)
    for i, classifier in enumerate(classifiers):
        y_pred[:, i] = classifier.predict(X_test_scaled)

    # Evaluation
    print('\n\nMultilabel Classification Evaluation \n\n')
    print("Hamming Loss:", hamming_loss(y_test_cls, y_pred))
    print("Precision (micro):", precision_score(y_test_cls, y_pred, average='micro'))
    print("Recall (micro):", recall_score(y_test_cls, y_pred, average='micro'))
    print("F1 Score (micro):", f1_score(y_test_cls, y_pred, average='micro'))
    print("Precision (macro):", precision_score(y_test_cls, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_test_cls, y_pred, average='macro'))
    print("F1 Score (macro):", f1_score(y_test_cls, y_pred, average='macro'))

    return True