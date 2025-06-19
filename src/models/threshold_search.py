import pandas as pd 
import numpy as np  
from src.models.utils import * 

from src.config import(
    DATA_PATHS,
    SEARCH_THRESHOLD_LOG_DIR, 
    THRESHOLD_SEARCH_PATH, 
    
)

def search_threshold(d,path_to_save):
    
    distances = []
    thresholds  = get_thresholds_from_proba(d["predicted_proba"])
    print(thresholds)
        
    # get hcc patients
    df = d[d["age_hepatocellular_carcinoma_dp_dr"].notna()]

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
    distances_df.to_csv(f"{path_to_save}/wasserstein_distance_by_threshold.csv", index=False) 
    plot_age_hcc(d, path_to_save, thresholds)
    
    
def search_best_seil(model_name, logger): 
    # load the data : 
    logger.info(f"Loading data with predictions")
    data_with_predictions = load_data(f'{DATA_PATHS["data_with_predictions"]}/{model_name}') 
    
    # search threshold      
    pth = THRESHOLD_SEARCH_PATH / model_name 
    pth.mkdir(parents=True, exist_ok=True)
    search_threshold(data_with_predictions["all_data"],pth)


if __name__ == "__main__":
    #parse the model arguments
    args = parse_model_args()
    # setup logger 
    logger = setup_logger(SEARCH_THRESHOLD_LOG_DIR, args.model)
    search_best_seil(args.model,logger)
