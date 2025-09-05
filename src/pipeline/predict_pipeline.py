import sys
import pandas as pd
from src.exception import OurException
from src.logger import logging
from src.utils import load_obj

class PredictPipeline():
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred 

        except Exception as e:
            logging.info("Excption Occured in Prediction Pipeline")
            raise OurException(e, sys)
        
class CustomData:
    def __init__(self,
        N : int,
        P : int,
        K : int,
        temperature : float,
        humidity : float,
        ph : float,
        rainfall : float):
        
        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "N" : [self.N],
                "P" : [self.P],
                "K" : [self.K],
                "temperature" : [self.temperature],
                "humidity" : [self.humidity],
                "ph" : [self.ph],
                "rainfall" : [self.rainfall]
            }

            return pd.DataFrame(custom_data_input_dict) 

        except Exception as e:
            raise OurException(e, sys)
