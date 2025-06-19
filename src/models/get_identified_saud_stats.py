import pandas as pd 
import numpy as np  
from src.models.utils import * 
from src.models.available_models import models


from src.config import(
    DATA_PATHS, 
    IDENTIFICATION_SAUD_PATH, 
    IDENTIFICATION_SAUD_LOG_PATH,
    FEATURE_GROUPS
)


def plot_age_hcc_optimal_plot(data_dict, xgb_data_with_predictions, svm_data_with_predictions, mlp_data_with_predictions, age_variable, seuil_xg, seuil_mlp, seil_svm):
    """
    Plots the distribution of age at HCC onset for different groups:
    - all T2D patients
    - patients with AUD
    - patients predicted as SAUD by XGBoost, MLP, and SVM models.

    Also computes the Wasserstein distance between the predicted SAUD distributions 
    and the actual AUD group, saving the results to a CSV file.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the raw input datasets keyed by phase. 
    xgb_data_with_predictions : dict
        Dictionary containing XGBoost prediction results keyed by phase.
    svm_data_with_predictions : dict
        Dictionary containing SVM prediction results keyed by phase.
    mlp_data_with_predictions : dict
        Dictionary containing MLP prediction results keyed by phase.
    age_variable : str
        Name of the column representing age at HCC onset.
    seuil_xg : float
        Threshold for XGBoost predicted probabilities to define SAUD.
    seuil_mlp : float
        Threshold for MLP predicted probabilities to define SAUD.
    seil_svm : float
        Threshold for SVM predicted probabilities to define SAUD.

    Saves
    -----
    - Age distribution plot in EPS format.
    - Wasserstein distance values in CSV file.
    """
    
    distance = []
    
    for set_name, d in data_dict.items():
        if(set_name =="all_data"):
            df = d[d[age_variable].notna()]
            df_mlp = mlp_data_with_predictions[set_name][mlp_data_with_predictions[set_name][age_variable].notna()]
            df_svm = svm_data_with_predictions[set_name][svm_data_with_predictions[set_name][age_variable].notna()]
            df_xgb = xgb_data_with_predictions[set_name][xgb_data_with_predictions[set_name][age_variable].notna()]
        
            saud_mlp = df_mlp[(df_mlp['alcohol_use_disorders'] == 0) & (df_mlp['predicted_proba'] >= seuil_mlp)]
            saud_xg  = df_xgb[(df_xgb['alcohol_use_disorders'] == 0) & (df_xgb['predicted_proba'] >= seuil_xg)]
            saud_svm = df_svm[(df_svm['alcohol_use_disorders'] == 0) & (df_svm['predicted_proba']>=seil_svm )]

            fig, ax = plt.subplots(figsize=(8, 6))  

            sns.kdeplot(
                data=df, x=age_variable, 
                color="black", fill=False, common_norm=False, 
                alpha=0.5, linewidth=2, label="all T2D"
            )
                
            sns.kdeplot(
                    data=df[df["alcohol_use_disorders"] == 1], 
                    x=age_variable, color="orange", fill=False, 
                    common_norm=False, alpha=0.5, linewidth=2, label=f"AUD"
            )
                
                    
            sns.kdeplot(
                    data=saud_xg, x=age_variable, 
                    color="blue", fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2
            )
            sns.kdeplot(
                    data=saud_mlp, x=age_variable, 
                    color="green", fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2
            )
            sns.kdeplot(
                    data=saud_svm, x=age_variable, 
                    color="red", fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2
            )


            plt.xlabel("Age at HCC onset")  
            plt.ylabel("Distribution")  
            plt.tight_layout()
            
            save_path = f"{IDENTIFICATION_SAUD_PATH}/age_distribution_{set_name}.eps"
            plt.savefig(save_path, dpi=300, format="eps")
            
            #------------ get wasseretin distance : 
            aud = df[df["alcohol_use_disorders"] == 1]
            dist_xg = kde_wasserstein_distance(saud_xg[age_variable], aud[age_variable]) 
            dist_svm = kde_wasserstein_distance(saud_svm[age_variable], aud[age_variable]) 
            dist_mlp = kde_wasserstein_distance(saud_mlp[age_variable], aud[age_variable]) 

            distance.append({
                "phase": set_name, 
                "dis_xg": dist_xg , 
                "dis_svm": dist_svm , 
                "dis_mlp": dist_mlp}
            ) 
            distance = pd.DataFrame(distance)
            distance.to_csv(f"{IDENTIFICATION_SAUD_PATH}/dist_wasserteIn_plots.csv", index=False) 
   
   
def get_sauds_stats(data_dict, data_with_predictions, model_threshold_pairs): 
    """
    Computes statistics on SAUD patients identified by different models, including:
    - number and proportion of SAUDs,
    - ratio of SAUDs to AUD and non-AUD populations,
    - proportion of SAUDs with HCC onset information.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the original dataset keyed by phase.
    data_with_predictions : dict
        Dictionary where each key is a model name and value is another dict containing predictions per phase.
    model_threshold_pairs : dict
        Dictionary mapping model names to threshold values for SAUD classification.

    Saves
    -----
    - A CSV file summarizing statistics of SAUD detection per model.
    """
    for set_name in ["all_data"]:  
        res = []
        nb_phase = len(data_dict[set_name])
        nb_non_aud = len(data_dict[set_name][data_dict[set_name]["alcohol_use_disorders"]==0])
        nb_aud =     len(data_dict[set_name][data_dict[set_name]["alcohol_use_disorders"]==1])

        for model_name, threshold in model_threshold_pairs.items():
            df= data_with_predictions[model_name][set_name]
            saud  = df[(df['alcohol_use_disorders'] == 0) & (df['predicted_proba'] >= threshold)]
            saud_hcc = saud[saud["age_hepatocellular_carcinoma_dp_dr"].notna()]
        
            res.append({
            "Model": model_name,
            "nb_saud": len(saud), 
            "%saud ": round(len(saud)/nb_phase*100,3), 
            "%saud/aud": round(len(saud)/nb_aud*100,3), 
            "%saud/non_aud": round(len(saud)/nb_non_aud*100,3), 
            "prev_hcc": round(len(saud_hcc)/len(saud)*100,3), 
                
            })

        res = pd.DataFrame(res)
        res.to_csv(f"{IDENTIFICATION_SAUD_PATH}/saud_stats_by_model.csv", index=False)


def get_saud_stats(logger, model_threshold_pairs): 
    """
    Main evaluation function to:
    - Load data and model predictions,
    - Plot age distribution comparison for HCC onset,
    - Generate and save SAUD statistics.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to track execution and debugging information.
    model_threshold_pairs : dict
        Dictionary with model names as keys and thresholds as values.
    """
    
    #Load the data with the predictions of all the evailable models
    logger.info(f"Load data With predictions")

    data = load_data(DATA_PATHS["processed"])  
    # Load model predictions
    models_data_with_predictions = {model_name: load_data(F'{DATA_PATHS["data_with_predictions"]}/{model_name}') for model_name in model_threshold_pairs}
    
    logger.info(f"Plot the optimal age at HCC onset with different models ")
    #plot_age_hcc_optimal_plot(data, models_data_with_predictions["xgboost"], models_data_with_predictions["oneClassSVM"], models_data_with_predictions["MLP"], FEATURE_GROUPS["hcc_age_onset"], model_threshold_pairs["xgboost"], model_threshold_pairs["oneClassSVM"], model_threshold_pairs["MLP"])
    logger.info(f"get stats on the number of saud / saud with HCC  ")
    get_sauds_stats(data, models_data_with_predictions, model_threshold_pairs)



if __name__ == "__main__":
    
    #parse the model arguments
    # setup logger 
    logger = setup_logger(IDENTIFICATION_SAUD_LOG_PATH, 'identified_saud_logs')
    
    args = parse_model_args(require_threshold=True)
    model_threshold_pairs = dict(zip(args.model, args.threshold)) 
    logger.info(f"Evaluating identified saud with {model_threshold_pairs}")
    get_saud_stats(logger, model_threshold_pairs)
