import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import joblib

def train(training_data, validation_data = None, random_state = 2025):
    # training_data = {
    #     'X_train': X_train_scaled,
    #     'y_cls_train': y_cls_train,
    #     'y_rgr_train': y_rgr_train_scaled
    # }

    X_train = training_data['X_train']
    y_cls_train = training_data['y_cls_train']

    checkpoint_path = "checkpoints"
    
    # Step 5: Initialize SVM with OneVsRestClassifier
    print("Loading Decision Tree Classifier")
    dt_classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state))
    dt_classifier.fit(X_train, y_cls_train)
    print(dt_classifier.get_params())
    
    # Save the model
    joblib.dump(dt_classifier, os.path.join(checkpoint_path,"decision_tree_classifier.pkl"))

    print('Training Successful!')

    return True

def evaluate(test_data, scaler_X, scaler_y, class_names, checkpoint_path = 'checkpoints'):
  X_test = test_data['X_test']
  y_test_cls = test_data['y_cls_test']

  # standard scaling input X
  X_test_scaled = scaler_X.transform(X_test.values)

  # Step 6: Make predictions
  dt_classifier = joblib.load(os.path.join(checkpoint_path,"decision_tree_classifier.pkl"))
  y_pred = dt_classifier.predict(X_test_scaled)

  # Step 7: Evaluation
  print('\n\nClassification Evaluation \n\n')
  print("Hamming Loss:", hamming_loss(y_test_cls, y_pred))
  print("Accuracy Score (exact match ratio):", accuracy_score(y_test_cls, y_pred))
  print("\nClassification Report:\n", classification_report(y_test_cls, y_pred, target_names=class_names))
  return True
  

