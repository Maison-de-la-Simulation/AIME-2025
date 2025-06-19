
import pandas as pd 
import numpy as np  
from pathlib import Path
from src.models.utils import * 


from src.config import( 
    PREDICT_LOG_DIR_PATH, 
    DATA_PATHS,
    DATA_WITH_PEDICTIONS_PATH
)

def save_prediction(data_with_preds,model_name): 
    
    pth = DATA_WITH_PEDICTIONS_PATH / model_name 
    pth.mkdir(parents=True, exist_ok=True)
    
    for set_name, d in data_with_preds.items(): 
        d.to_csv(f"{pth}/{set_name}.csv", index=False) 



def predict_model(model_name,logger): 
    
    # Initialize the model class
    logger.info(f"Model Initialization")
    model_class = models[model_name]
    traine_model = model_class(logger=logger)
    # load trained model
    traine_model.load_model() 
    
    # load the preprocessed data : 
    logger.info(f"Loading data")
    data = load_data(DATA_PATHS["processed"]) 
    
    logger.info(f"Split features and target data")
    XS,YS =  split_features_target(data)
    
    #Preprocess HCC variable
    logger.info(f"process hcc variabl ")
    pre_process_age_variables(XS)
    
    # get proba prediction 
    logger.info(f"Calculate predictions")

    for set_name, d in XS.items():
        preds_proba = traine_model.predict_classification_proba(d)
        preds_label = traine_model.predict(d)
        data[set_name] = pd.concat([data[set_name].reset_index(drop=True), preds_proba.reset_index(drop=True), preds_label.reset_index(drop=True)], axis=1) 
        
    # Save the model predictions 
    logger.info(f"Save predictions")
    save_prediction(data,model_name)
    
   

if __name__ == "__main__":
    
    #parse the model arguments
    args = parse_model_args()
    # setup logger 
    logger = setup_logger(PREDICT_LOG_DIR_PATH, args.model )
    
    predict_model(args.model,logger)

