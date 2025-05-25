import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial import ConvexHull

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback

import joblib
from models.multihead_earlystopping import MultiHeadEarlyStopping

def build_ann(input_dim=28, n_classes=4):
    input_layer = Input(shape=(input_dim,))

    # Shared base network
    x = Dense(128, activation='relu')(input_layer)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)

    # Classification head
    classification_branch = Dense(64, activation='relu')(x)
    # classification_branch = BatchNormalization()(classification_branch)
    classification_branch = Dropout(0.3)(classification_branch)
    classification_output = Dense(n_classes, activation='sigmoid', name='classification_head')(classification_branch)

    # Regression head
    regression_branch = Dense(64, activation='relu')(x)
    regression_branch = BatchNormalization()(regression_branch)
    regression_branch = Dropout(0.3)(regression_branch)
    regression_output = Dense(n_classes, activation='linear', name='regression_head')(regression_branch)

    # Create model
    model = Model(inputs=input_layer, outputs=[classification_output, regression_output])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'classification_head': 'binary_crossentropy', 'regression_head': 'mse'}
                  # metrics={'classification_head': 'accuracy', 'regression_head': 'mae'}
                  )

    return model

def train(training_data, validation_data, optimal_k = 4, random_state = 2025, patience = 10, epochs = 200, batch_size = 32):
    # training_data = {
    #     'X_train': X_train_scaled,
    #     'y_cls_train': y_cls_train,
    #     'y_rgr_train': y_rgr_train_scaled
    # }
    # validation_data = {
    #     'X_val': X_val_scaled,
    #     'y_cls_val': y_cls_val,
    #     'y_rgr_val': y_rgr_val_scaled
    # }
    # optimal_k = 4 ## The optimal number of clusters 
    # load_checkpoints = False ## If load model parameters from saved checkpoints

    X_train = training_data['X_train']
    y_cls_train = training_data['y_cls_train']
    y_rgr_train = training_data['y_rgr_train']

    X_val = validation_data['X_val']
    y_cls_val = validation_data['y_cls_val']
    y_rgr_val = validation_data['y_rgr_val']

    checkpoint_path = "/content"
    
    # Get multiheaded ann model
    print('Loading ANN Model...')
    model = build_ann()

    #Load custom early stopping function
    early_stop = MultiHeadEarlyStopping(patience=10)
    
    # Train the model
    print('Training ANN...')
    history = model.fit(
        X_train,
        {'classification_head': y_cls_train, 'regression_head': y_rgr_train},
        validation_data=(X_val, {'classification_head': y_cls_val, 'regression_head': y_rgr_val}),
        epochs=200,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(checkpoint_path, 'history.csv'), index=False)

    # Save the model
    model.save(os.path.join(checkpoint_path, 'ann_model.keras'))

    print('Training Successful!')

    return True


def evaluate(test_data, scaler_X, scaler_y, class_names, checkpoint_path = '/content/'):
  X_test = test_data['X_test']
  y_test_cls = test_data['y_cls_test']
  y_test_rgr = test_data['y_rgr_test']

  # standard scaling input X
  X_test_scaled = scaler_X.transform(X_test.values)

  model = tf.keras.models.load_model(os.path.join(checkpoint_path, 'ann_model.keras'))
  y_pred_cls, y_pred_rgr_scaled = model.predict(X_test_scaled)
  
  y_pred_rgr = scaler_y.inverse_transform(y_pred_rgr_scaled)
  y_pred_cls = (y_pred_cls > 0.5).astype(int)


  # Evaluation for regression Task
  print('Regression Evaluation\n')
  for i, target in enumerate(class_names):
      mse = mean_squared_error(y_test_rgr.iloc[:, i], y_pred_rgr[:, i])
      r2 = r2_score(y_test_rgr.iloc[:, i], y_pred_rgr[:, i])
      print(f"\nTarget: {target}")
      print(f"Mean Squared Error: {mse:.4f}")
      print(f"R2 Score: {r2:.4f}")

  
  # Evaluation for classification task
  print('\n\nClassification Evaluation \n\n')
  print("Hamming Loss:", hamming_loss(y_test_cls, y_pred_cls))
  print("Accuracy Score (exact match ratio):", accuracy_score(y_test_cls, y_pred_cls))
  print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls, target_names=class_names, zero_division= 1.0))

  return True
  