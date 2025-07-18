
import pandas as pd 
import numpy as np  
import sys 
from pathlib import Path
from tqdm import tqdm
from src.models.utils import * 
from sklearn.model_selection import ParameterGrid

# add the project root to the path 
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import( 
    DATA_PATHS,
    TRAIN_LOG_DIR_PATH, 
    get_hyperparams_grid, 
    selected_features, 
    model_hyperparam_selection_criterion,
    
)


def search_best_model(XS, YS, model_name, logger): 
    """
    Performs hyperparameter search to find the best model configuration.

    This function iterates over a grid of hyperparameters, trains a model for each configuration,
    evaluates it on the validation set, and returns the best model based on he AUC criterion.

    Parameters
    ----------
    XS : dict
        Dictionary containing data for 'train' and 'validation'.
    YS : dict
        Dictionary containing labels for 'train' and 'validation'.
        
    model_name : str
        The name of the model to train (used to retrieve the class and hyperparameter grid).
        
    logger : logging.Logger 
        Logger instance used to log the training steps.
    
    Returns
    -------
    best_model : 
        The trained model instance with the best validation performance.
    """
    report = [] 
    trained_models = [] 
    
    user_params = get_hyperparams_grid(model_name)
    paramset = list(ParameterGrid(user_params)) 
    
    for params in tqdm(paramset, total=len(paramset)):    
        model = models[model_name](model_params=params, logger=logger)
        
        model.train(XS, YS) 
        
        # predict the outputs of the trained model using the validation data 
        y_hat =  model.predict(XS["validation"])
    
        # get the classification performances on the validation data 
        perfs = get_performances(YS["validation"], y_hat , "validation")
    
        report.append(perfs)
        trained_models.append(model)
        
    # get the best model based in a metric: here we use the auc criterio 
    rapport_df  = pd.DataFrame.from_dict(report)  
    idx_best_model = rapport_df[model_hyperparam_selection_criterion].idxmax() 
    
    return trained_models[idx_best_model] 


def train_model(model_name, logger):
    """
    Train the specified model using hyperparameter search.

    This function checks the validity of the model name, loads the data,
    performs hyperparameter search to find the best configuration, and saves the best model.

    Parameters
    ----------
    model_name : str
        The name of the model to train. Must be a key in the global `models` dictionary.

    logger : Logger 
    """ 
    
    logger.info(f"Load data")

    # load the training data 
    data = load_data(DATA_PATHS["processed"]) 
    print(data.head)
    data = data[selected_features]
    print(data.head)
    
    logger.info(f"Split data into X:features  and Y:target ")
    # get the features data and the target data. . 
    XS,YS =  split_features_target(data)

    logger.info(f"Preprocess HCC variable ")
    # Hcc variable preprocessing : 
    pre_process_age_variables(XS)
    
    logger.info(f"Training {model_name} model with hyperparameter search ! ")
    best_model = search_best_model(XS, YS, model_name, logger)
    
    # save the best model 
    model_path = best_model.save()
    logger.info(f"Model training completed. Best Model saved at {model_path}")
    

if __name__ == "__main__":
    #parse the model arguments    
    args = parse_model_args()
    # setup logger
    print(args.model)
    logger = setup_logger(TRAIN_LOG_DIR_PATH, args.model)
    
    train_model(args.model,logger)




