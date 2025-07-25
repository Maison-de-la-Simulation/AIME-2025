
import pandas as pd 
import numpy as np  
import sys 
from pathlib import Path
from tqdm import tqdm
from src.models.utils import * 
from sklearn.model_selection import ParameterGrid
import os 

# add the project root to the path 
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import( 
    DATA_PATHS,
    TRAIN_LOG_DIR_PATH, 
    get_hyperparams_grid, 
    MODELS_PATHS, 
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
        y_hat_label =  model.predict(XS["validation"])
        y_hat_proba =  model.predict_classification_proba(XS["validation"])
    
        # get the classification performances on the validation data 
        perfs = get_performances(YS["validation"], y_hat_label , y_hat_proba, "validation")
    
        report.append(perfs)
        trained_models.append(model)
        
    # get the best model based in a metric: here we use the auc criterio 
    rapport_df  = pd.DataFrame.from_dict(report) 
    # sauvegarder le rapport_df: 
    rapport_df.to_csv(f"{TRAIN_LOG_DIR_PATH}/rapport_perfs_auc_all_models.csv", index=False)
    
    # get the best model: 
    max_crit_val = rapport_df[model_hyperparam_selection_criterion].max()
    candidats = rapport_df[rapport_df[model_hyperparam_selection_criterion] == max_crit_val]
    idx_best_model = candidats["tpr"].idxmax()
    
    #idx_best_model = rapport_df[model_hyperparam_selection_criterion].idxmax()
    best_hyperparam = paramset[idx_best_model]
    best_model = trained_models[idx_best_model] 
    
    return idx_best_model, best_hyperparam, best_model, trained_models, rapport_df

def save_best_trained_model(logger, best_model,model_path): 
    best_model.save(model_path) 
    logger.info(f"Model training completed. Best Model saved at {model_path}") 


def train(logger, model_name, XS, YS ): 
    logger.info(f"Training {model_name} model with hyperparameter search ! ")
    idx_best_model, best_hyperparam, best_model, trained_models, rapport_df = search_best_model(XS, YS, model_name, logger)
    return idx_best_model, best_hyperparam, best_model, trained_models, rapport_df 
    

def train_model(model_name, logger, path_to_save):
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
    
    data = get_data(logger, DATA_PATHS["processed"])
    data = select_features(data)
    XS,YS = prepare_training_data(logger, data)
    idx_best_model, best_hyperparam, best_model, trained_models, rapport_df  = train(logger, model_name, XS, YS )
    save_best_trained_model(logger, best_model, path_to_save)
    
    
if __name__ == "__main__":
    #parse the model arguments    
    args = parse_model_args()
    logger = setup_logger(TRAIN_LOG_DIR_PATH, args.model)
    path_to_save = os.path.join(MODELS_PATHS[args.model], f"best_{args.model}.pkl") 
    train_model(args.model,logger, path_to_save)




