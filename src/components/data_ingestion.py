import os
import src
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sys
from src.components.data_transformation import DataTrannsformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainingConfig,ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifact','train.csv')
    test_data_path=os.path.join('artifact','test.csv')
    raw_data_path=os.path.join('artifact','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiated_data_ingestion(self):
        logging.info('starting data ingestion....')
        try:
            df=pd.read_csv('E:/Machine Learning/E2E-AWS-Benstalk/notebook/data/stud.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train test intiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Injestion completed')
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path,
                    )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    traindata,testdata= obj.initiated_data_ingestion()
    obj_datatrans=DataTrannsformation()
    train_arr,test_arr,_=obj_datatrans.initiate_data_trasnformation(traindata,testdata)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr))
