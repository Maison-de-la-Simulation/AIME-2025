import pandas as pd 
import numpy as np  
from src.models.utils import * 

from src.config import(
    DATA_PATHS,
    SEARCH_THRESHOLD_LOG_DIR, 
    THRESHOLD_SEARCH_PATH, 
    
)


def search_threshold(logger,model_name, d):
    
    distances = []
    logger.info(f"get list of thresholds")
    thresholds  = get_thresholds_from_proba(d["predicted_proba"])
    print(thresholds)
    # get hcc patients
    df = d[d["age_hepatocellular_carcinoma_dp_dr"].notna()]

    logger.info(f"Get distances by threshold")
    for i, threshold in enumerate(thresholds):
        all_sauds = d[(d['alcohol_use_disorders'] == 0) & (d['predicted_proba'] >= threshold)]
        nb_all_sauds = len(all_sauds)
        if(nb_all_sauds!=0): 
            
            #hcc_sauds = df[(df['alcohol_use_disorders'] == 0) & (df['predicted_proba'] >= threshold)]
            hcc_sauds = all_sauds[all_sauds['age_hepatocellular_carcinoma_dp_dr'].notna()]
            nb_saud_hcc =  len(hcc_sauds) 
            
            # calculer la distance des deux distribution: 
            distance = kde_wasserstein_distance(df[df["alcohol_use_disorders"] == 1]["age_hepatocellular_carcinoma_dp_dr"],hcc_sauds["age_hepatocellular_carcinoma_dp_dr"])
            if(distance):
                distances.append({
                    "seil": threshold,
                    "dist": round(distance, 2),
                    "nb_saud" : nb_all_sauds, 
                    "%saud_/total" : round(nb_all_sauds/len(d) *100,2),
                    "nb_saud_hcc" : nb_saud_hcc, 
                    "saud_hcc/saud" : round(nb_saud_hcc/nb_all_sauds *100, 2)
                })    
    distances_df = pd.DataFrame(distances)
    
    logger.info(f"Save distances dataframe")
    distances_df.to_csv(f"{THRESHOLD_SEARCH_PATH[model_name]}/wasserstein_distance_by_threshold.csv", index=False) 
    
    logger.info(f"plot AHO by threshold")
    plot_age_hcc(d, THRESHOLD_SEARCH_PATH[model_name], thresholds)
    
    # get the optimal threshold : 
    best_threshold = distances_df.loc[distances_df["dist"].idxmin(), "seil"]   
    return best_threshold 
    
    
    
def search_best_seil(model_name, logger): 
    # load the data : 
    data_with_predictions = get_data(logger, f'{DATA_PATHS["data_with_predictions"]}/{model_name}') 

    # search threshold      
    best_threshold = search_threshold(logger, model_name,  data_with_predictions["all_data"]) 
    print(best_threshold)


if __name__ == "__main__":
    #parse the model arguments
    args = parse_model_args()
    # setup logger 
    logger = setup_logger(SEARCH_THRESHOLD_LOG_DIR, args.model)
    search_best_seil(args.model,logger)
