import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

if __name__ == '__main__':
    # configs
    training = False
    model_name = 'ann'

    if model_name == 'ann_with_clustering':
        from models.ann_with_clustering import train, evaluate
    elif model_name == 'ann':
        from models.ann import train, evaluate
    elif model_name == 'support_vector_classifier':
        from models.support_vector_classifier import train, evaluate
    elif model_name == 'random_forest_classifier':
        from models.random_forest_classifier import train, evaluate
    elif model_name == 'decision_tree_classifier':
        from models.decision_tree_classifier import train, evaluate
    elif model_name == 'knn_classifier':
        from models.knn_classifier import train, evaluate  
    elif model_name == 'xgb_classifier':
        from models.xgb_classifier import train, evaluate  
    elif model_name == 'support_vector_regressor':
        from models.support_vector_regressor import train, evaluate
    elif model_name == 'random_forest_regressor':
        from models.random_forest_regressor import train, evaluate
    elif model_name == 'decision_tree_regressor':
        from models.decision_tree_regressor import train, evaluate
    elif model_name == 'knn_regressor':
        from models.knn_regressor import train, evaluate
    elif model_name == 'xgb_regressor':
        from models.xgb_regressor import train, evaluate


    if training:
        train_df = pd.read_csv('dataset/train_data2.csv')
        
        #Preparing dataset
        X = train_df.loc[:, 'Q0':'Q27']
        y = train_df[['Visual', 'Auditorial', 'Reading', 'Kinesthetic']]

        # classification output
        y_cls = (y > 0.5).astype(int)

        # regression output
        y_rgr = y.copy()

        X_train = X.values[0: int((X.shape[0])*0.8)]
        y_cls_train = y_cls.values[0: int((X.shape[0])*0.8)]
        y_rgr_train = y_rgr.values[0: int((X.shape[0])*0.8)]

        X_val = X.values[int((X.shape[0])*0.8):]
        y_cls_val = y_cls.values[int((X.shape[0])*0.8):]
        y_rgr_val = y_rgr.values[int((X.shape[0])*0.8):]


        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        scaler_y = StandardScaler()
        scaler_y.fit(y_rgr_train)
        y_rgr_train_scaled = scaler_y.transform(y_rgr_train)
        y_rgr_val_scaled = scaler_y.transform(y_rgr_val)

        # joblib.dump(scaler, '/content/scaler_X.pkl')
        # joblib.dump(scaler_y, '/content/scaler_y.pkl')

        training_data = {
                'X_train': X_train_scaled,
                'y_cls_train': y_cls_train,
                'y_rgr_train': y_rgr_train_scaled
            }
        validation_data = {
                'X_val': X_val_scaled,
                'y_cls_val': y_cls_val,
                'y_rgr_val': y_rgr_val_scaled
            }
        train(training_data, validation_data)

    else:
        test_df = pd.read_csv('dataset/test_data2.csv')
        X_test = test_df.loc[:, 'Q0':'Q27']
        X_test = X_test.fillna(0.5)
        y_test = test_df[['Visual', 'Auditorial', 'Reading', 'Kinesthetic']]
        y_test_rgr = y_test.copy()
        y_test_cls = (y_test > 0.5).astype(int)

        test_data = {
            'X_test': X_test,
            'y_cls_test': y_test_cls,
            'y_rgr_test': y_test_rgr
        }

        scaler_X = joblib.load('checkpoints/scaler_X.pkl')
        scaler_y = joblib.load('checkpoints/scaler_y.pkl')

        evaluate(test_data, scaler_X, scaler_y, y_test.columns)


    