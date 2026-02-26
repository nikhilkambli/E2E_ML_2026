#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor
import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    trained_model_path=os.path.join('artifact','model.pkl')


class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('spliting training and test input data')
            xtrain,ytrain,xtest,ytest=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model={

                "Random Forest":RandomForestRegressor(),
                "Decision TreeRegressor":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "Gradient Boost":GradientBoostingRegressor(),
                "adaboost":AdaBoostRegressor(),
                "k_NEIGHBOURS":KNeighborsRegressor()

            }

            model_report:dict=evaluate_models(X_train=xtrain,y_train=ytrain,X_test=xtest,y_test=ytest,models=model)

            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print("best model name ",best_model_name)

            best_model=model[best_model_name]
            print("best_model ",best_model)

            if best_model_score<0.6:
                raise CustomException('No best fit model')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model

            )

            predicted=best_model.predict(xtest)
            r2_square=r2_score(ytest,predicted)
            return r2_square




        except Exception as e:
            raise CustomException(e,sys)