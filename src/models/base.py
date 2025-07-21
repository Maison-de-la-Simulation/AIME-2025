from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib
import os 

from src.config import(
    MODELS_PATHS 
)

class BaseModel(ABC):
    """
    Abstract base class for all models in the project.
    """
    
    def __init__(self,name,logger, model_params=None):
        """
        Initializes the base model.
        Parameters
        ----------
        name : str
            Name of the model. 
        model_params : dict
            Dictionary of model-specific hyperparameters.
        logger : logging.Logger 
            Logger instance used to log 
    
        """
        self.model = None
        self.name = name 
        self.model_params = model_params 
        self.logger = logger
        
        
    @abstractmethod
    def train(self, XS, YS):
        """
        Abstract method for training the model.

        Parameters
        ----------
        XS : dict of pd.DataFrame
            Dictionary of data sets by phase (e.g., train, validation).
        YS : dict of pd.Series
            Dictionary of labels/targets by phase.
        """
        pass
    
    
    @abstractmethod
    def predict(self, X):
        """
        Abstract method for making predictions.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data for prediction.

        Returns
        -------
        array-like
            Predictions made by the model.
        """
        pass
    
    @abstractmethod
    def predict_classification_proba(self, X):
        """
        Abstract method for making predictions probabilities.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data for prediction.

        Returns
        -------
        array-like
            Predictions made by the model.
        """
        pass
    

    def save(self, model_path):
        """
        Saves the trained model to disk using joblib.

        Returns
        -------
        str
            The file path where the model was saved.
        """
        joblib.dump({
                     "model": self.model,
                     "params": self.model_params
                    }, model_path)
        
        
    def load_model(self, model_path):
        """
        Loads a previously saved trained model from disk.

        Returns
        -------
        BaseModel
            The model instance with the loaded model.
        """
        loaded_obj = joblib.load(model_path)
        self.model = loaded_obj["model"]
        self.model_params = loaded_obj["params"]
        

    





