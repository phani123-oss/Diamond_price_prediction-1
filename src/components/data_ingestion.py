import os ##for creating path i.e, train test
import sys #for logging and exception
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

#Initialise DataIngestion Configuration
from dataclasses import dataclass
@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv') #Train path inside artifacts folder
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

#Class data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('DataIngestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data/gemstone.csv'))
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Raw data is created')

            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

            
        except Exception as e:
            logging.info('Exception Occurs at DataIngestion stage')
            raise CustomException(e,sys)


