

from src.models.train_model import *
from src.models.predict_model import * 
from src.models.evaluate_classification_perfs import *
from src.models.threshold_search import * 
from src.models.get_saud_stats import *

from src.models.utils import * 
from plot_functions import *
from src.models.available_models import models

from src.config import(
    ITERATIVE_LEARNING_LOG_PATH, 
    NB_iterations_by_model, 
    MODELS_PATHS, 
    IDENTIFICATION_SAUD_PLOTS_PATH
)

 
def do_one_iteration(logger,model_name,iteration_number,XS, YS, data): 

    # get the best trained model : 
    idx_best_model, best_hyperparam, best_model, trained_models, rapport_df  = train(logger, model_name, XS, YS)
    print(best_hyperparam)
    print(idx_best_model)
    print(rapport_df)
    
    # save the model : 
    path_to_save = os.path.join(MODELS_PATHS[model_name], f"best_{model_name}_iteration_{iteration_number}.pkl") 
    save_best_trained_model(logger,best_model,path_to_save)
        
    # get the predictions with the best found model: 
    data_with_predictions = calculate_predictions(logger, best_model, XS, data) 
        
    # get the model perfermances :
    dataframe_perf = calculate_classif_perfs(logger, data_with_predictions) 
    save_classification_performances_csv(dataframe_perf,os.path.join(CLASSIF_PERFS_DIR_PATH[model_name], f"{model_name}_iteration_{iteration_number}.csv"))
    
    # get the AHO distances by threshold 
    distances_df = search_threshold_iterative_learning(logger,model_name, data_with_predictions["all_data"])
    save_distances(logger, model_name, distances_df, f"{THRESHOLD_SEARCH_PATH[model_name]}/wass_dist_by_threshold_iteration_{iteration_number}.csv")
        
    # get the best threshol 
    best_threshold = get_best_threshold(distances_df)
        
    # get saud stats by the best thresthol 
    stats = get_stats(logger, data_with_predictions, best_threshold)
        
    # save the stats : 
    stats.to_csv(f"{IDENTIFICATION_SAUD_PATH[model_name]}/saud_stats_{model_name}_itaration_{iteration_number}.csv", index=False)
    
    return data_with_predictions , best_threshold 
    
def update_with_saud(logger, data_with_predictions, best_threshold): 
    
    logger.info(f"Update AUD with identified sAUD")
    for set_name , df in data_with_predictions.items(): 
        df["alcohol_use_disorders"] = ((df["alcohol_use_disorders"] == 1) | (df["predicted_proba"] > best_threshold)).astype(int)
        data_with_predictions[set_name] = df 
        
    return data_with_predictions 
    
def iterative_train_model(model_name, logger):
    
    logger.info(f"Start iterative learning")
   
    data = get_data(logger, DATA_PATHS["processed"])
    data_history = get_data(logger, DATA_PATHS["processed"]) 
    #create the initial_aud feature : 
    for set_name, df in data.items():
        df['initial_aud'] = df['alcohol_use_disorders'].copy()
        data[set_name]=df
            
    for i in range(NB_iterations_by_model[model_name]): 
        logger.info(f"******** Iterative number {i} *********")
        
        data = select_features(data) 
        XS, YS = prepare_training_data(logger, data)
        data_with_predictions, best_threshold = do_one_iteration(logger,model_name,i,XS, YS, data)
        # stocker les proba des differents iteration : 
        storage(logger, data_history, data_with_predictions, i,best_threshold)
        save_data_with_predictions(data_history, logger, model_name )
        
        # just-qu'a la dans data_with_predictions y a les prediction : donc je dois mettre les saud dans les aud pour la prochaine iteration : 
        data = update_with_saud(logger, data_with_predictions, best_threshold)
    
    # AHO plots 
    plot_1(data_history, model_name, IDENTIFICATION_SAUD_PLOTS_PATH[model_name]) 
    plot_2(data_history, model_name, IDENTIFICATION_SAUD_PLOTS_PATH[model_name])
    plot_3(data_history, model_name, IDENTIFICATION_SAUD_PLOTS_PATH[model_name])
        
        
 

        
if __name__ == "__main__":
    
    args = parse_model_args()
    # setup logger
    print(args.model)
    logger = setup_logger(ITERATIVE_LEARNING_LOG_PATH, args.model)
    iterative_train_model(args.model,logger)






