# import project modules  
from src.models.xgboost import XGBoostModel
from src.models.oneClassSVM import OneClassSvmModel 
from src.models.MLP import MLPModel 

models = {
    'xgboost': XGBoostModel,
    'oneClassSVM': OneClassSvmModel,
    'MLP': MLPModel
}
