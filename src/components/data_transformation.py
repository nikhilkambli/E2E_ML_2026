#Feature Engineering,Data Cleaning
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTrannsformation:

    def __init__(self):
        self.data_trannsformation_config=DataTransformationConfig()
    
    def get_data_trnsformation_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                 ]
              
            num_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )
            logging.info('Catogorical columns encoding completed')
#ColumnTransformer used to apply transformation on diffrent columns and then combine result into one feature matrxi(works parreler )
#and if we use num piple her then cat piplin appiled to output of numerical  piplne as its sequantial 
            prrprocessor=ColumnTransformer(
                [
                   ("numpipline",num_pipline,numerical_columns),
                   ("catPiple",cat_pipline,categorical_columns) 
                ]
            )

            return prrprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_trasnformation(self,train_path,test_path):
         try:
             train_df=pd.read_csv(train_path)
             test_df=pd.read_csv(test_path)
             logging.info("Reading of training and test data completed")
             logging.info('obtaining preprocoor')
             preprocessing_obj=self.get_data_trnsformation_object()
             target_column_name="math score"
             numerical_columns = ["writing_score", "reading_score"]
             input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
             target_feature_train_df=train_df[target_column_name]

             #Testing data
             input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
             target_feature_test_df=test_df[target_column_name]

             logging.info("applying proprocessing on training and test")

             input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
             input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)
#concatination
             train_array=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
             test_array=np.c_[input_feature_test_array,np.array(target_feature_test_df)]

             save_object(
                   file_path=self.data_trannsformation_config.preprocessor_obj_file_path,
                   obj=preprocessing_obj
                 )
             
             return (
                     train_array,
                     test_array,
                     preprocessing_obj
                  )

         
         except Exception as e:
             raise CustomException(e,sys)






        
            
