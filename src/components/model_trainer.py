import os
import sys
from src.exception import OurException
from src.logger import logging
from src.utils import evaluate_models, save_obj

from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

from sklearn import metrics

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data.")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Decision Tree" : DecisionTreeClassifier(),
                "SVC" : SVC(),
                "Gaussian" : GaussianNB(),
                "Random Forest" : RandomForestClassifier(),
                "Logistic" : LogisticRegression(),
                "K Neighbors" : KNeighborsClassifier(),
                "Cat Boost" : CatBoostClassifier(verbose=False),
            }

            params = {
                "Decision Tree" : {
                  "criterion" : ['gini', 'entropy', 'log_loss'],
                  "splitter" : ['best', 'random'],
                  "max_features" : ['sqrt', 'log2']  
                },
                "SVC" : {
                    "kernel" : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    "gamma" : ['scale', 'auto']
                },
                "Gaussian" : {
                    'gnb__var_smoothing': [1e-9, 1e-10, 1e-11]
                },
                "Random Forest" : {
                  "criterion" : ['gini', 'entropy', 'log_loss'],
                  "class_weight" : ['balanced', 'balanced_subsample'],
                  "max_features" : ['sqrt', 'log2', None]
                },
                "Logistic" : {
                    "penalty" : ['l1', 'l2', 'elasticnet'],
                    "solver" : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                },
                "K Neighbors" : {
                    'n_neighbors': range(1, 21),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Cat Boost" : {
                    'iterations': [30, 50, 100],
                    'learning-rate' : [0.01, 0.05, 0.1],
                    'depth': [6,8,10]
                }
            }

            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise OurException("No Best Model Found")
            
            logging.info("Best Model found on both training and test dataset.")

            save_obj(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = metrics.accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise OurException(e, sys)