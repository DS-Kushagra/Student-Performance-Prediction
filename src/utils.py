import os
import sys
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import dill

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, param):
    try:
        report = {}
        for i, model_name in enumerate(models.keys()):
            model = models[model_name]  # Get the model instance
            para = param[model_name]  # Get the parameters for this model

            # Perform Grid Search
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Use the best estimator from Grid Search
            best_model = gs.best_estimator_

            # Train the model
            best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # R^2 Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in the report
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)