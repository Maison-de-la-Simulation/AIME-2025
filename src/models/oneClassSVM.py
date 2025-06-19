import numpy as np 
import pandas as pd 
from src.models.base import BaseModel
from sklearn.svm import OneClassSVM


class OneClassSvmModel(BaseModel):
    """One-class SVM classifier model"""
    
    def __init__(self,logger, model_params=None): 
        super().__init__(name="oneClassSVM",model_params=model_params, logger=logger)
        if model_params is not None : 
            self.model = OneClassSVM(**self.model_params)
        else : 
            self.model = None 
            self.model_params = {}
     
    def train(self, XS, YS):
        """Train the One-Class SVM model."""
        X_train, Y_train = XS["train"], YS["train"]
        # select the data of the positive class 
        aud_train_data = X_train[Y_train==1] 
        self.model.fit(aud_train_data)
        return self 

    def predict(self, X):
        """Make predictions with the One-class SVM model."""
        y_hat = self.model.predict(X)  
        y_hat= np.where(y_hat== -1, 0,1)  
        return pd.DataFrame(y_hat, columns=['predicted_label']) 
    
    
    def predict_classification_proba(self, X):
        """
        Calculates the distance to the learned decision boundary with the trained OneClass SVM model.
        """
        prediction_proba = pd.DataFrame(self.model.decision_function(X), columns=['predicted_proba'])
        return prediction_proba['predicted_proba']
         
        
  