import os
import sys
from src.exception import OurException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered in data ingestion component.")
        try:
            crop_data = pd.read_csv("notebook/data/crop_recommendation_dataset.csv")
            logging.info("Read the dataset as dataframe successfully.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            crop_data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test Split initiated.")

            train_set, test_set = train_test_split(crop_data, test_size=0.2, random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed succesfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise OurException(e, sys)

if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()