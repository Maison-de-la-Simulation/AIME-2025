

from src.models.train_model import *
from src.models.predict_model import * 
from src.models.evaluate_classification_perfs import *
from src.models.threshold_search import * 
from src.models.get_saud_stats import *

from src.models.utils import * 
from src.models.available_models import models

from src.config import(
    ITERATIVE_LEARNING_LOG_PATH, 
    NB_iterations_by_model, 
    MODELS_PATHS, 
    IDENTIFICATION_SAUD_PLOTS_PATH
)


def get_union_saud(df, num_iteration ): 
    # renvoie tout les patient qui ete classifier comme saud au moin une seule fois jusqu'a l'iteration num_iteration 
    
    colonnes_cibles = [f"sAUD_{i}" for i in range(num_iteration+1)]
    masque = df[colonnes_cibles].eq(1).any(axis=1) 
    return df[masque]

def get_non_aud_pure(df, num_iteration): 
    non_aud = df[df["alcohol_use_disorders"] == 0]
    colonnes_cibles = [f"sAUD_{i}" for i in range(num_iteration+1)]
    masque = non_aud[colonnes_cibles].eq(0).all(axis=1) 
    return non_aud[masque]


def  plot_3(data_history, model_name): 
    # on plot juste les vrai aud, l'union des saud identifier jusqu'à l'iteration courante, et les non-aud moin l'union des saud 
    for set_name, dat in data_history.items(): 
        data_HCCs = dat[dat["age_hepatocellular_carcinoma_dp_dr"].notna()] 
        fig, ax = plt.subplots(figsize=(8, 6))  
        # Tracé de la distribution des AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 1], 
            x="age_hepatocellular_carcinoma_dp_dr", color="black", fill=False, 
            common_norm=False, alpha=0.5, linewidth=2, label=f"AUD"
        )
        
         # Tracé de la distribution des non_AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 0], 
            x="age_hepatocellular_carcinoma_dp_dr", color="black", fill=False, 
            linestyle="--", common_norm=False, alpha=0.5, linewidth=2, label=f"non-AUD"
        )
    
        iterations = NB_iterations_by_model[model_name]
        palette = sns.color_palette("husl", iterations)
        palette_non_aud = sns.color_palette("flare", iterations)

        for i in range(iterations):
            saud = get_union_saud(data_HCCs, i)
            pure_non_aud = get_non_aud_pure(data_HCCs, i)
            nb_saud = len(saud) 
            nb_non_aud_pure = len(pure_non_aud)
            if(nb_saud!=0) :  
                sns.kdeplot(
                    data=saud, x="age_hepatocellular_carcinoma_dp_dr", 
                    color=palette[i], fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2, 
                    label=f"Union_sAUD_jsq{i}"
                )
                
            if(nb_non_aud_pure!=0) :  
                sns.kdeplot(
                    data=pure_non_aud, x="age_hepatocellular_carcinoma_dp_dr", 
                    color=palette_non_aud[i], fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2, 
                    linestyle="--",
                    label=f"non_AUD_jsq{i}"
                )
                
        # Configuration des axes et titre
        ax.set_title(f"Distribution de age_hepatocellular_carcinoma_dp_dr set={set_name})", fontsize=14, pad=15)
        ax.grid(visible=True, linestyle="--", alpha=0.5)
                
        # Ajouter une légende
        ax.legend(loc="best", fontsize=10)
        save_path = f"{IDENTIFICATION_SAUD_PLOTS_PATH[model_name]}/AHO_pure_nonAUD_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   




def  plot_2(data_history, model_name): 
    # on plot juste les vrai aud, l'union des saud identifier jusqu'à l'iteration courante, et les non-aud de base 
    for set_name, dat in data_history.items(): 
        data_HCCs = dat[dat["age_hepatocellular_carcinoma_dp_dr"].notna()] 
        fig, ax = plt.subplots(figsize=(8, 6))  
        # Tracé de la distribution des AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 1], 
            x="age_hepatocellular_carcinoma_dp_dr", color="orange", fill=False, 
            common_norm=False, alpha=0.5, linewidth=2, label=f"AUD"
        )
        # Tracé de la distribution des non-AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 0], 
            x="age_hepatocellular_carcinoma_dp_dr", color="blue", fill=False, 
            common_norm=False, alpha=0.5, linewidth=2, label=f"AUD"
        )
        
        iterations = NB_iterations_by_model[model_name]
        palette = sns.color_palette("husl", iterations)
        
        for i in range(iterations):
            saud = get_union_saud(data_HCCs, i)
            nb_saud = len(saud) 
            if(nb_saud!=0) :  
                sns.kdeplot(
                    data=saud, x="age_hepatocellular_carcinoma_dp_dr", 
                    color=palette[i], fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2, 
                    label=f"Union_sAUD_jsq{i}"
                )
                
        # Configuration des axes et titre
        ax.set_title(f"Distribution de age_hepatocellular_carcinoma_dp_dr set={set_name})", fontsize=14, pad=15)
        ax.grid(visible=True, linestyle="--", alpha=0.5)
                
        # Ajouter une légende
        ax.legend(loc="best", fontsize=10)
        save_path = f"{IDENTIFICATION_SAUD_PLOTS_PATH[model_name]}/AHO_union_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   
       

def  plot_1(data_history, model_name): 
    # plot 1: on plot juste les vrai aud, les saud identifier dans l'iteration courante, et les non-aud de base 
    for set_name, dat in data_history.items(): 
        data_HCCs = dat[dat["age_hepatocellular_carcinoma_dp_dr"].notna()] 
        fig, ax = plt.subplots(figsize=(8, 6))  
        # Tracé de la distribution des AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 1], 
            x="age_hepatocellular_carcinoma_dp_dr", color="orange", fill=False, 
            common_norm=False, alpha=0.5, linewidth=2, label=f"AUD"
        )
        # Tracé de la distribution des non-AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 0], 
            x="age_hepatocellular_carcinoma_dp_dr", color="blue", fill=False, 
            common_norm=False, alpha=0.5, linewidth=2, label=f"non-AUD"
        )
        
        iterations = NB_iterations_by_model[model_name]
        palette = sns.color_palette("husl", iterations)
        
        for i in range(iterations):
            saud = data_HCCs[data_HCCs[f"sAUD_{i}"]==1]
            nb_saud = len(saud) 
            if(nb_saud!=0) :  
                sns.kdeplot(
                    data=saud, x="age_hepatocellular_carcinoma_dp_dr", 
                    color=palette[i], fill=False, common_norm=False, 
                    alpha=0.5,
                    linewidth=2, 
                    label=f"sAUD_{i}"
                )
                
        # Configuration des axes et titre
        ax.set_title(f"Distribution de age_hepatocellular_carcinoma_dp_dr set={set_name})", fontsize=14, pad=15)
        ax.grid(visible=True, linestyle="--", alpha=0.5)
                
        # Ajouter une légende
        ax.legend(loc="best", fontsize=10)
        save_path = f"{IDENTIFICATION_SAUD_PLOTS_PATH[model_name]}/AHO_disjoint_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   
       
        
    

def do_one_iteration(logger,model_name,iteration_number,XS, YS, data): 

    # get the best trained model : 
    idx_best_model, best_hyperparam, best_model, trained_models, rapport_df = train(logger, model_name, XS, YS )
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
    distances_df = search_threshold(logger,model_name, data_with_predictions["all_data"])
    save_distances(logger, model_name, distances_df, f"{THRESHOLD_SEARCH_PATH[model_name]}/wass_dist_by_threshold_iteration_{iteration_number}.csv")
        
    # get the best threshol 
    best_threshold = get_best_threshold(distances_df)
        
    # get saud stats by the best thresthol 
    stats = get_stats(logger, data_with_predictions, best_threshold)
        
    # save the stats : 
    stats.to_csv(f"{IDENTIFICATION_SAUD_PATH[model_name]}/saud_stats_{model_name}_itaration_{iteration_number}.csv", index=False)
    
    return data_with_predictions , best_threshold 
    

def update_with_saud(logger, data_with_predictions,best_threshold): 
    
    logger.info(f"Update AUD with identified sAUD")
    for set_name , df in data_with_predictions.items(): 
        df["alcohol_use_disorders"] = ((df["alcohol_use_disorders"] == 1) | (df["predicted_proba"] > best_threshold)).astype(int)
        data_with_predictions[set_name] = df 
        
    return data_with_predictions 
    
def storage(logger, data_history, data_with_predictions, iteration_number, best_threshold): 
    logger.info(f"Storage iteration {iteration_number} results")
    for set_name , df in data_history.items():    
        df[f"predicted_proba_{iteration_number}"] = data_with_predictions[set_name]["predicted_proba"]
        df[f"predicted_label_{iteration_number}"] = data_with_predictions[set_name]["predicted_label"]
        df[f"sAUD_{iteration_number}"] = ((data_with_predictions[set_name]["alcohol_use_disorders"] == 0) & (data_with_predictions[set_name]["predicted_proba"] > best_threshold)).astype(int)
        data_history[set_name] = df 
        

def iterative_train_model(model_name, logger):
    
    logger.info(f"Start iterative learning")
   
    data = get_data(logger, DATA_PATHS["processed"])
    data_history = get_data(logger, DATA_PATHS["processed"]) 
    
    for i in range(NB_iterations_by_model[model_name]): 
        logger.info(f"******** Iterative number {i} *********")
        
        data = select_features(data) 
        XS, YS = prepare_training_data(logger, data)
        data_with_predictions, best_threshold = do_one_iteration(logger,model_name,i,XS, YS, data)
        # stocker les proba des differents iteration : 
        storage(logger, data_history, data_with_predictions, i,best_threshold)
        
        # just-qu'a la dans data_with_predictions y a les prediction : donc je dois mettre les saud dans les aud pour la prochaine iteration : 
        data = update_with_saud(logger, data_with_predictions, best_threshold )
    
    # la il faut penser à stocjer data_history comme data_with prediction comme avant 
    # AHO plots 
    plot_1(data_history, model_name) 
    plot_2(data_history, model_name)
    plot_3(data_history, model_name)
        
        
        

        
if __name__ == "__main__":
    
    args = parse_model_args()
    # setup logger
    print(args.model)
    logger = setup_logger(ITERATIVE_LEARNING_LOG_PATH, args.model)
    iterative_train_model(args.model,logger)







