
import pandas as pd 
from src.models.utils import * 

from src.config import( 
    PREDICT_LOG_DIR_PATH, 
    MODELS_PATHS, 
    DATA_WITH_PEDICTIONS_PATH
)

def save_prediction(data_with_preds,model_name): 
    for set_name, d in data_with_preds.items(): 
        d.to_csv(f"{DATA_WITH_PEDICTIONS_PATH[model_name]}/{set_name}.csv", index=False) 



def load_model(model_name,logger, model_path): 
    
    # Initialize the model class
    logger.info(f"Model Initialization")
    model_class = models[model_name]
    traine_model = model_class(logger=logger)
    # load trained model
    traine_model.load_model(model_path) 
    
    return traine_model 

    
def calculate_predictions(logger, traine_model, XS, data): 
     # get proba prediction 
    logger.info(f"Calculate predictions")

    for set_name, d in XS.items():
        preds_proba = traine_model.predict_classification_proba(d)
        preds_label = traine_model.predict(d)
        data[set_name] = pd.concat([data[set_name].reset_index(drop=True), preds_proba.reset_index(drop=True), preds_label.reset_index(drop=True)], axis=1) 
        
    return data 

def save_data_with_predictions(data, logger,model_name ): 
    logger.info(f"Save predictions")
    save_prediction(data,model_name)
    
    
def predict_model(model_name,logger, model_path ): 
    
    
    # best trained model load 
    traine_model = load_model(model_name,logger, model_path) 
    
    # load the data 
    data = get_data(logger, DATA_PATHS["processed"])
    data = select_features(data)
    # prepare the data : 
    XS,YS = prepare_training_data(logger,data )

    # get the predictions : 
    data_with_predictions = calculate_predictions(logger, traine_model, XS, data)
    
    # Save the model predictions 
    save_data_with_predictions(data_with_predictions, logger, model_name)
    
   
if __name__ == "__main__":
    
    #parse the model arguments
    args = parse_model_args()
    # setup logger 
    logger = setup_logger(PREDICT_LOG_DIR_PATH, args.model)
    model_path = os.path.join(MODELS_PATHS[args.model], f"best_{args.model}.pkl") 
    predict_model(args.model,logger, model_path)

