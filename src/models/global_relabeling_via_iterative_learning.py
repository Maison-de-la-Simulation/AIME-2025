

from src.models.train_model import *
from src.models.predict_model import * 
from src.models.evaluate_classification_perfs import *
from src.models.threshold_search import * 
from src.models.get_saud_stats import *

from src.models.utils import * 
from src.models.plot_functions import *
from src.models.available_models import models

from src.config import(
    GLOBAL_LABELING_LOG_PATH, 
    NB_iterations_by_model, 
    MODELS_PATHS, 
    DATA_WITH_FINAL_PEDICTIONS_PATH, 
    IDENTIFICATION_FINAL_SAUD_PLOTS_PATH, 
    best_threshold_par_iteration, 
    GLOBAL_ITERATIF_CLASSIF_DIR_PATH, 
    IDENTIFICATION_FINAL_SAUD_STATS_PATH
)


def predict_final_saud(model_name, logger, best_thresholds): 
    data = get_data(logger, DATA_PATHS["processed"])  
    data_history = get_data(logger, DATA_PATHS["processed"]) 

    for set_name, df in data.items():
        df['initial_aud'] = df['alcohol_use_disorders'].copy()
        data[set_name]=df
        
    for i in range(NB_iterations_by_model[model_name]): 
        model_name_i = f"best_{model_name}_iteration_{i}"
        logger.info(f"******** Get final sAUDs with the {model_name_i} *********")

        data = select_features(data) 
        XS, YS = prepare_training_data(logger, data) 
    
        logger.info(f"******** Load {model_name_i} ********")
        model_i_path = f"{MODELS_PATHS[model_name]}/{model_name_i}.pkl"
        model_i = load_model(model_name,logger, model_i_path) 
        
        data = calculate_predictions(logger, model_i, XS, data) 
        
        perfs_models_i = calculate_classif_perfs(logger, data)
        save_perfs_path = os.path.join(GLOBAL_ITERATIF_CLASSIF_DIR_PATH[model_name], f"{model_name_i}_conf_matrice.csv") 
        save_classification_performances_csv(perfs_models_i, save_perfs_path)
        
        storage(logger, data_history, data, i, best_thresholds[i])
        
    logger.info(f"********  Save the data with predictions  ********")
    for set_name, d in data_history.items(): 
        d.to_csv(f"{DATA_WITH_FINAL_PEDICTIONS_PATH[model_name]}/{set_name}.csv", index=False) 

    return data_history 

    


def global_labeling(model_name, logger): 
    logger.info(f"********  get final sAUDs through global labeling with the iterative models  ********")
    data__with_predictions = predict_final_saud(model_name, logger, best_threshold_par_iteration[model_name])
    logger.info(f"********  Plot the golobal sAUDs through global labeling with the iterative models  ********")
    
    # AHO plots 
    plot_1(data__with_predictions, model_name, IDENTIFICATION_FINAL_SAUD_PLOTS_PATH[model_name]) 
    plot_2(data__with_predictions, model_name, IDENTIFICATION_FINAL_SAUD_PLOTS_PATH[model_name])
    plot_3(data__with_predictions, model_name, IDENTIFICATION_FINAL_SAUD_PLOTS_PATH[model_name])
    plot_intersection_saud(data__with_predictions, model_name, IDENTIFICATION_FINAL_SAUD_PLOTS_PATH[model_name])    
    plot_saud_intersection_nAUD_pure(data__with_predictions, model_name, IDENTIFICATION_FINAL_SAUD_PLOTS_PATH[model_name])

if __name__ == "__main__":
    
    args = parse_model_args()
    # setup logger
    print(args.model)
    logger = setup_logger(GLOBAL_LABELING_LOG_PATH, args.model)
    global_labeling(args.model,logger)


