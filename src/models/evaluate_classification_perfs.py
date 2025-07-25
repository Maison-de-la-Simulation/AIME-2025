
import pandas as pd 
import numpy as np  
import sys 
from pathlib import Path
from src.models.utils import * 

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import(
    EVAL_LOG_DIR_PATH, 
    DATA_PATHS
)

def calculate_classif_perfs(logger, data_with_predictions): 
    
    logger.info(f"Calculate classification performances")
    perfs = []
    for set_name, d in data_with_predictions.items():
        perfs.append(get_performances(d[FEATURE_GROUPS["target_variable"]], d["predicted_label"], d["predicted_proba"], set_name))
        
    dataframe_perf = pd.DataFrame(perfs)
    return dataframe_perf


def evaluate_model(model_name, logger): 
    """
    Evaluate a trained model by loading it, making predictions, 
    computing classification performances, and saving the results.

    Parameters
    ----------
    model_name : str
        The name of the model to evaluate. It must correspond to a key in the `models` dictionary.
    
    logger : logging.Logger
        Logger instance used to log messages during the evaluation process.
    """

    # load the data : 
    data_with_predictions = get_data(logger, f'{DATA_PATHS["data_with_predictions"][model_name]}') 
    
    dataframe_perf = calculate_classif_perfs(logger, data_with_predictions)
    
    save_path = os.path.join(CLASSIF_PERFS_DIR_PATH[model_name], f"{model_name}_conf_matrice.csv") 
    save_classification_performances_csv(dataframe_perf,save_path)
    logger.info(f"{model_name} Classification Performances saved at: {save_path}")


if __name__ == "__main__":
    #parse the model arguments
    args = parse_model_args()
     # setup logger 
    logger = setup_logger(EVAL_LOG_DIR_PATH, args.model)  
    
    evaluate_model(args.model, logger) 