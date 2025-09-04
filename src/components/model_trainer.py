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
            }

            params = {
                "Decision Tree" : {
                  "criterion" : ['gini', 'entropy'],
                  "max_depth" : [5, 10, 20, None],
                  "min_samples_split" : [2, 5, 10],
                  "max_features" : ['sqrt', 'log2', None]  
                },
                "SVC" : {
                    "kernel" : ['linear', 'poly', 'rbf'],
                    "C" : [0.1, 1, 10],
                    "gamma" : ['scale', 'auto'],
                    "degree" : [2, 3] #Only Used for 'poly' kernel 
                },
                "Gaussian" : {
                    'var_smoothing': [1e-9, 1e-10]
                },
                "Random Forest" : {
                  "n_estimators" : [50, 100],  
                  "criterion" : ['gini', 'entropy'],
                  "max_depth" : [10, 20, None],
                  "max_features" : ['sqrt', 'log2'],
                  "class_weight" : ['balanced']
                },
                "Logistic" : {
                    "penalty" : ['l2'],
                    "C" : [0.1, 1, 10],
                    "solver" : ['lbfgs', 'liblinear', 'saga']
                },
                "K Neighbors" : {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ['uniform', 'distance'],
                    "metric": ['euclidean', 'manhattan']
                }
            }

            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

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