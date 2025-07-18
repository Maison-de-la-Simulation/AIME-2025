
import pandas as pd 
import numpy as np 
from pathlib import Path
import logging
import argparse
import scipy.stats as stats 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from src.models.available_models import models
from src.features.preprocessing import pre_process_age_variables


from src.config import( 
    FEATURE_GROUPS, 
    CLASSIF_PERFS_DIR_PATH,
    REQUIRED_PARAMS
)


def setup_logger(log_file_path: Path, name: str): 
    """
    Set up and configure a logger that outputs logs both to the console and to a file.

    Parameters
    ----------
    log_file_path : Path
        The directory path where the log file will be saved.
    
    name : str
        The name of the logger instance.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    print(log_file_path)
    log_dir = log_file_path / f"{name}.log"
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logging    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir)
        ]
    )   
    logger = logging.getLogger(name)
    return logger 


def load_data(path): 
    """
    Load the data for training, validation, test, and all combined phases.
    
    Parameters : 
    ----------
    
    path : path
        The path to the data to lead 
        
    Returns
    -------
    XS : dict of pd.DataFrame
        Dictionary containing data for each phase: 'train', 'validation', 'test', 'all_data'.
    """
    data = dict()
    
    for phase in ["train","validation","test","all_data"]:   
        data[phase] = pd.read_csv(f"{path}/{phase}.csv", low_memory=False)

    return data 


def split_features_target(data): 
    XS = {}
    YS = {}
    
    for set_name, data_phase in data.items():  
        XS[set_name]  = data_phase.drop(columns=FEATURE_GROUPS["target_variable"]) 
        YS[set_name]  = data_phase[FEATURE_GROUPS["target_variable"]]

    return  XS ,YS 



def load_data_for_uncertainty_estimation(path): 
    """
    Load the data to use to estimate the uncertainty of the models
    
    Parameters : 
    ----------
    
    path : path
        The path to the data to lead 
        
    Returns
    -------
    XS : dict of pd.DataFrame
        Dictionary containing data for each phase: 'train', 'validation', 'test', 'all_data' and train_val. 
    """

    data = load_data(path) 
    XS, YS =  split_features_target(data)
    
    #concatener train and validation data. 
    XS["train_val"] = pd.concat([XS["train"], XS["validation"]], axis=0).reset_index(drop=True)
    YS["train_val"] = pd.concat([YS["train"], YS["validation"]], axis=0).reset_index(drop=True)

    return XS, YS 
    
def get_performances(YS, YS_hat, set_name ): 
    """
    Compute classification performance metrics for a model.
    Parameters
    ----------
    y : pd.Series
        True labels of the data (binary classification: 0 and 1).
    
    y_hat : pd.Series
        Predicted labels or probabilities for the data (binary classification: 0 and 1).
        
    set_name : str
        the name of the data set (train, val, test)
    Returns
    -------
    dict
        A dictionary containing various classification metrics:
    """
    accuracy = accuracy_score(YS,YS_hat)
    conf_matrix = confusion_matrix(YS,YS_hat)
    tn, fp, fn, tp = conf_matrix.ravel()
    auc = roc_auc_score(YS,YS_hat)
        
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
        
    f1 = f1_score(YS,YS_hat)
    
    return {
        "set_name" : set_name,
        "auc" : round(auc,2),
        "acc" : round(accuracy,2),
        "fpr ": round(fpr,2),
        "tnr" : round(tnr,2),
        "fnr" : round(fnr,2),
        "tpr" : round(tpr,2),
        "F1 score" : round(f1,2)
    }
        
    
def save_classification_performances_csv(perfs, model_name):
    """
    Save classification performance metrics to a CSV file.

    Parameters
    ----------
    perfs : pd.DataFrame
        DataFrame containing the classification performance metrics to save.
    
    model_name : str
        The name of the model, used to name the saved CSV file.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.
    """
    save_path = CLASSIF_PERFS_DIR_PATH / f"{model_name}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    perfs.to_csv(save_path, index=False)
    return save_path 


def validate_params(model_name, params):
    required = REQUIRED_PARAMS.get(model_name, [])
    missing = [param for param in required if param not in params]
    if missing:
        raise ValueError(f"Missing required hyperparameters for {model_name}: {missing}")


def parse_models_with_multiple_args(require_threshold=False, require_hyperparam=False):
    """
    Parse command-line arguments to select a model for evaluation.

    Parameters
    ----------
    require_threshold : bool
        Whether the threshold argument is required.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the selected model name.

    Raises
    ------
    ValueError
        If the provided model name is not in the available models.
    """
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(models.keys()),
                        help='Model to evaluate')
    
    parser.add_argument('--threshold', type=float, nargs='+' ,required=require_threshold,
                        help='Threshold value for decision')
    
    parser.add_argument('--hyperparam_file', type=str, required=require_hyperparam,
                        help='Model hyperparameters as JSON file')


    args = parser.parse_args()  
    
    if require_threshold and len(args.model) != len(args.threshold):
        raise ValueError("You must provide one threshold per model.")

    for model in args.model:
        if model not in models:
            raise ValueError(f"Unknown model: {model}. Available models: {list(models.keys())}")
    
    return args


def parse_model_args(require_threshold=False, require_hyperparam=False):
    """
    Parse command-line arguments to select a model for evaluation.

    Parameters
    ----------
    require_threshold : bool
        Whether the threshold argument is required.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the selected model name.

    Raises
    ------
    ValueError
        If the provided model name is not in the available models.
    """
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(models.keys()),
                        help='Model to evaluate')
    
    parser.add_argument('--threshold', type=float,required=require_threshold,
                        help='Threshold value for decision')
    
    parser.add_argument('--hyperparam_file', type=str, required=require_hyperparam,
                        help='Model hyperparameters as JSON file')


    args = parser.parse_args()  
    
 
    return args


def kde_wasserstein_distance(data1, data2, num_points=1000):
    """
    Computes the Wasserstein distance between two distributions estimated via Kernel Density Estimation (KDE).
    Parameters 
    ---------- 
    data1 : array-like  
        First dataset used to estimate the density.  
    data2 : array-like
        Second dataset used to estimate the density. 
    num_points : int, optional (default=1000)
        Number of points used to discretize the KDEs on a common grid.
    Returns
    -------
    float 
        Wasserstein distance between the two estimated distributions. 
    """   
    if(len(data1)>1 and len(data2)>1): 
        kde1 = stats.gaussian_kde(data1) 
        kde2 = stats.gaussian_kde(data2) 
    
        x_min = min(data1.min(), data2.min()) 
        x_max = max(data1.max(), data2.max()) 
        x_grid = np.linspace(x_min, x_max, num_points) 

        pdf1 = kde1(x_grid)  
        pdf2 = kde2(x_grid) 
    
        return wasserstein_distance(x_grid, x_grid, pdf1, pdf2)
    

def get_thresholds_from_proba(d ): 
    
    max_value = 1 ##max(d)
    
    low_density_values = np.round(np.linspace(0.5, 0.8*max_value, 10, endpoint=False),3)  
    high_density_values = np.round(np.linspace(0.8*max_value, 0.96*max_value, 50, endpoint=False),3)  
    few_high_values = np.round(np.linspace(0.96*max_value, max_value, 10, endpoint=True),3)
    thresholds = np.sort(np.concatenate((low_density_values, high_density_values, few_high_values)))
    return thresholds
 

def plot_age_hcc(d, path_to_save, thresholds):
    
    df = d[d["age_hepatocellular_carcinoma_dp_dr"].notna()]
    
    with PdfPages(f"{path_to_save}/plot_hcc_age_onset_by_seuils.pdf") as pdf:
        for threshold in thresholds:
            subset = df[(df['alcohol_use_disorders'] == 0) & (df['predicted_proba'] >= threshold)]
            nb_saud = len(subset) 
            if(nb_saud!=0) : 
                fig, ax = plt.subplots(figsize=(8, 6))  
                # Tracé de la distribution générale
                sns.kdeplot(
                        data=df, x="age_hepatocellular_carcinoma_dp_dr", 
                        color="grey", fill=False, common_norm=False, 
                        alpha=0.5, linewidth=2, label="all T2D"
                )
                # Tracé de la distribution des AUD
                sns.kdeplot(
                        data=df[df["alcohol_use_disorders"] == 1], 
                        x="age_hepatocellular_carcinoma_dp_dr", color="orange", fill=False, 
                        common_norm=False, alpha=0.5, linewidth=2, label=f"AUD"
                )
                
                sns.kdeplot(
                        data=subset, x="age_hepatocellular_carcinoma_dp_dr", 
                        color="blue", fill=False, common_norm=False, 
                        alpha=0.5,
                        linewidth=2, 
                        label=f"sAUD (Seuil={threshold:.2f})"
                )
                
                # Configuration des axes et titre
                ax.set_title(f"Distribution de age_hepatocellular_carcinoma_dp_dr (Seuil={threshold:.2f})", fontsize=14, pad=15)
                ax.grid(visible=True, linestyle="--", alpha=0.5)
                
                # Ajouter une légende
                ax.legend(loc="best", fontsize=10)
                
                # Sauvegarder la figure dans le fichier PDF
                pdf.savefig(fig)  # Sauvegarder la figure actuelle dans le PDF
                plt.close(fig)  # Fermer l


def get_stat_saud(d, threshold): 
    all_sauds = d[(d['alcohol_use_disorders'] == 0) & (d['predicted_proba'] >= threshold)]
    nb_all_sauds = len(all_sauds)
    
    df = d[d["age_hepatocellular_carcinoma_dp_dr"].notna()]
    distance = {}
    if(nb_all_sauds!=0): 
        hcc_sauds = df[(df['alcohol_use_disorders'] == 0) & (df['predicted_proba'] >= threshold)]
        nb_saud_hcc =  len(hcc_sauds) 
            
        # calculer la distance des deux distribution: 
        distance = kde_wasserstein_distance(df[df["alcohol_use_disorders"] == 1]["age_hepatocellular_carcinoma_dp_dr"],hcc_sauds["age_hepatocellular_carcinoma_dp_dr"])
        if(distance):
            distance =  {
                    "dist": round(distance, 2),
                    "nb_saud" : nb_all_sauds, 
                    "%saud_/total" : round(nb_all_sauds/len(d) *100,2),
                    "nb_saud_hcc" : nb_saud_hcc, 
                    "saud_hcc/saud" : round(nb_saud_hcc/nb_all_sauds *100, 2)
                }
    return distance

    
    
    
    
    
    

