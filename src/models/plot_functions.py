from src.models.utils import * 
from src.models.available_models import models

from src.config import(
    NB_iterations_by_model, 
)


def  plot_3(data_history, model_name, path_to_save): 
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
        save_path = f"{path_to_save}/AHO_pure_nonAUD_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   


def  plot_intersection_saud(data_history, model_name, path_to_save): 
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
            saud = get_intersection_saud(data_HCCs, i)
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
        save_path = f"{path_to_save}/AHO_saud_intersection_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   

def  plot_2(data_history, model_name, path_to_save): 
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
        save_path = f"{path_to_save}/AHO_union_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   
       

def  plot_1(data_history, model_name, path_to_save): 
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
        save_path = f"{path_to_save}/AHO_disjoint_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   
    
    
def  plot_saud_intersection_nAUD_pure(data_history, model_name, path_to_save): 
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
        
         # Tracé de la distribution de base des non_AUD
        sns.kdeplot(
            data=data_HCCs[data_HCCs["alcohol_use_disorders"] == 0], 
            x="age_hepatocellular_carcinoma_dp_dr", color="black", fill=False, 
            linestyle="--", common_norm=False, alpha=0.5, linewidth=2, label=f"non-AUD"
        )
    
        iterations = NB_iterations_by_model[model_name]
        palette = sns.color_palette("husl", iterations)
        palette_non_aud = sns.color_palette("flare", iterations)

        for i in range(iterations):
            saud = get_intersection_saud(data_HCCs, i)
            pure_non_aud = get_non_aud_pure_intersection(data_HCCs, i)
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
        save_path = f"{path_to_save}/AHO_saud_inters_pure_nonAUD_{set_name}.png"
        plt.savefig(save_path, dpi=300, format="png")   

