from xgboost import XGBClassifier
from src.models.base import BaseModel
from collections import Counter
import pandas as pd 

class XGBoostModel(BaseModel):
    """XGBoost classifier model"""
    
    def __init__(self, logger,model_params=None):
        super().__init__(name="xgboost",model_params=model_params,logger=logger)
        if model_params is not None: 
            self.model = XGBClassifier(**self.model_params)
        else :  
            self.model = None 
            self.model_params = {}
            
    def train(self, Xs, Ys):
        """Train the XGBoost model."""
        # set the parameter scale_pos_weight=weights_xgboost
        counter = Counter(Ys["train"]) 
        weights_xgboost = counter[0]/counter[1]    
        self.model.set_params(scale_pos_weight=weights_xgboost) 
        self.model.fit(Xs["train"], Ys["train"]) 
        return self

    def train_cross_val(self,X_train_fold,y_train_fold,y_train): 
        # set the parameter scale_pos_weight=weights_xgboost
        counter = Counter(y_train) 
        weights_xgboost = counter[0]/counter[1]    
        self.model.set_params(scale_pos_weight=weights_xgboost) 
    
        self.model.fit(X_train_fold, y_train_fold)
        
        
        
        
    def predict(self, X):
        """Make predictions with the trained XGBoost model."""
        return pd.DataFrame(self.model.predict(X), columns=['predicted_label']) 
    
    
    
    def predict_classification_proba(self, X):
        """
        Make Probabilitie predictions with the trained XGBoost model.
        """
        prediction_proba = pd.DataFrame(self.model.predict_proba(X), columns=['proba_classe_0', 'predicted_proba'])
        return prediction_proba['predicted_proba']
         
    
    
    
    
    
    
    
    

 

 
 

        






