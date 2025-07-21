

from src.models.train_model import *
from src.models.predict_model import * 
from src.models.evaluate_classification_perfs import *
from src.models.threshold_search import * 
from src.models.get_saud_stats import get_saud_stats 

from src.models.utils import * 
from src.models.available_models import models

from src.config import(
    ITERATIVE_LEARNING_LOG_PATH, 
    NB_iterations_by_model, 
    MODELS_PATHS
)


def iterative_train_model(model_name, logger):
    
    perfs_over_itarations = {}
    
    logger.info(f"Start iterative learning")
    data = get_data(logger)
    XS, YS = prepare_training_data(logger, data)
    
    for i in range(NB_iterations_by_model[model_name]): 
        
        logger.info(f"Iterative number {i}")
        
        # get the best trained model : 
        best_model = train(logger, model_name, XS, YS )
        
        # save the model : 
        path_to_save = os.path.join(MODELS_PATHS[args.model], f"best_{model_name}_iteration_{i}.pkl") 
        save_best_trained_model(logger,best_model,path_to_save)
        
        # get the predictions with the best found model : 
        data_with_predictions = calculate_predictions(best_model, XS, data) 
        
        # get the model perfermances :
        dataframe_perf = calculate_classif_perfs(data_with_predictions) 

        # get the optimal threshold  
        best_threshold = search_threshold(logger,model_name, data_with_predictions["all_data"])
        print(best_threshold)
        
                
        

if __name__ == "__main__":
    
    args = parse_model_args()
    # setup logger
    print(args.model)
    logger = setup_logger(ITERATIVE_LEARNING_LOG_PATH, args.model)
    iterative_train_model(args.model,logger)







