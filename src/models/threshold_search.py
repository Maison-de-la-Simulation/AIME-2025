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
                
    logger.info(f"plot AHO by threshold")
    plot_age_hcc(d, THRESHOLD_SEARCH_PATH[model_name], thresholds)

    distances_df = pd.DataFrame(distances)
    return distances_df 


def save_distances(logger, model_name, distances_df, path_to_save): 
    logger.info(f"Save distances dataframe")
    distances_df.to_csv(path_to_save, index=False) 
    
    
def get_best_threshold(distances_df): 
    # get the optimal threshold : au lieu de faire ca je le saisie moi en se basant sur les resulat des distance wwassertein 
    #best_threshold = distances_df.loc[distances_df["dist"].idxmin(), "seil"]   
    while True:
        try:
            saisie = input("Veuillez entrer la valeur du threshold optimal ")
            best_threshold = float(saisie)
            break
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre valide.")
    
    return best_threshold
    
    
    
def search_best_seil(model_name, logger): 
    # load the data : 
    data_with_predictions = get_data(logger, f'{DATA_PATHS["data_with_predictions"][model_name]}') 

    # search threshold      
    distances_df = search_threshold(logger, model_name,  data_with_predictions["all_data"]) 
    
    # save the distance dataframe 
    save_distances(logger, model_name, distances_df, f"{THRESHOLD_SEARCH_PATH[model_name]}/wasserstein_distance_by_threshold.csv")
        
    best_threshold = get_best_threshold(distances_df)
    
    return best_threshold 

if __name__ == "__main__":
    #parse the model arguments
    args = parse_model_args()
    # setup logger 
    logger = setup_logger(SEARCH_THRESHOLD_LOG_DIR, args.model)
    best_threshold = search_best_seil(args.model,logger)
    print(f"best threshold ={best_threshold}")
