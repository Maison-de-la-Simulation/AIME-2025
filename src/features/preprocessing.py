import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold 
from tqdm import tqdm

from src.config import (
    DATA_PATHS,
    FEATURE_GROUPS,
    selected_features,
    nb_packets_split_data,
    hcc_ageOnset_fill_value,
    alpha
)


def get_patient_with_hcc_after_2013(data): 
    """
    Filters patients who developed hepatocellular carcinoma (HCC) after the year 2013.
    Parameters:
    ----------
    data : pd.DataFrame
        A DataFrame containing at least the following columns:
            - 'D_entry': date of entry into the system (format: 'YYYY-MM-DD')
            - 'age.min': the minimum age at which the patient had any recorded condition
    Returns:
    -------
    pd.DataFrame
        A filtered DataFrame containing only patients with HCC onset after 2013 
        or with no recorded HCC onset.
    """
    data['D_entry'] = pd.to_datetime(data['D_entry'], format='%Y-%m-%d')
    data["anne_hcc"] = data['D_entry'].dt.year + (data[FEATURE_GROUPS["hcc_age_onset"]] - data['age.min']) 
    data = data[(data["anne_hcc"].isna()) | (data["anne_hcc"] > 2013)]
    return data

def supprime_fdep15_Q5_na(data): 
    """
    Removes rows from the DataFrame where the 'fdep15_Q5' column contains missing values (NaN).

    Parameters:
    ----------
    data : pd.DataFrame
        A pandas DataFrame that must include a column named 'fdep15_Q5'.

    Returns:
    -------
    None
        The function does not return anything. It directly modifies the input DataFrame.
    """
    data.dropna(subset=['fdep15_Q5'], inplace=True)
    

def get_ages_variables(df): 
    """
    Return list of continuous variables (illness ages). 
    arameters:
        - df (pandas dataframe): the project data 
        
    Returns:
        - age_columns (list) : the age features in the data 

    """
    age_columns = [ col for col in df.columns if col.startswith('age') ]
    return age_columns


def OneHotEncoderForCategoricleColumns(df, categorical_columns):
    """
    Transforms categorical data into oneHotEncoding form
    Parameters:
        - df (pandas dataframe) : the project data 
        - categorical_columns (list) : list of categorical varaibles 
        
    Returns:
        - pandas dataframe: the data with the categorical variables transformed 

    """
    return pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)


def fill_na_ages_features(df, age_variables, fill_value) :
    """
    The function is used to impute missing values for age variables 
    Parameters:
        - df (pandas dataframe) : the project data 
        - age_variables (list) : List of continuous (age of onset of illness) variables 
        - fill_value (integer) : the value to fill na with 
    """
    
    df.loc[:,age_variables] = df[age_variables].fillna(fill_value)
 
 
def pre_process_age_variables(XS) : 
    
    for set_name, d in XS.items():
        ## fill the missing values 
        fill_na_ages_features(d,[FEATURE_GROUPS["hcc_age_onset"]], hcc_ageOnset_fill_value) 
        ## normalization of the variable
        normalize_age_colone(d,[FEATURE_GROUPS["hcc_age_onset"]],hcc_ageOnset_fill_value, alpha )


def normalize_age_colone(data, age_variables,fill_value, alpha ):
    """
    The function is used to normalize the age variables
    Parameters:
        - data (pandas dataframe) : the project data 
        - age_variables (list) : List of continuous (age of onset of illness) variables 
        - fill_value (integer) : the value to fill na with 
        - alpha (integer) : hyperparameter 
   
    """
    
    data.loc[:,age_variables] = ((((fill_value- data[age_variables])/fill_value)*alpha))  
   
def run_preprocessing_pipeline():
    
    steps = [
        "Reading raw data", 
        "Selecting patients with HCC after 2013", 
        "Deleting rows with NA in fdep15_Q5", 
        "One-hot encoding categorical variables", 
        "Selecting relevant features", 
        "Splitting data into train, validation, and test sets", 
        "Storing processed data"
    ]

    with tqdm(total=len(steps), ncols=100) as pbar:
    
        # read the raw data 
        data = pd.read_csv(DATA_PATHS["raw"], low_memory=False)
        pbar.update(1)
        
        # select patients with hcc developped after 2023
        data = get_patient_with_hcc_after_2013(data)
        pbar.update(1)
        
        # Deletes raw data with NA in the fdep15_Q5 variable.
        supprime_fdep15_Q5_na(data)
        pbar.update(1)
        
        # One hotEncoding of the categorical variable dfep15_q5 : 
        data = OneHotEncoderForCategoricleColumns(data, FEATURE_GROUPS["categorical_features"])
        pbar.update(1)
        
        # features extraction : in this research we use a subset of features 
        data = data[selected_features]  
        pbar.update(1)
        
        # split the data into packets
        skf = StratifiedKFold(n_splits=nb_packets_split_data["nb_packets"], shuffle=True, random_state=42)

        paquets = []
        for i, (train_index, test_index) in enumerate(skf.split(data, data[FEATURE_GROUPS["target_variable"]])):
            paquet = data.iloc[test_index]
            paquets.append(paquet) 

        #  get train/val/test sets : 
        train_data = pd.concat(paquets[:nb_packets_split_data["nb_packets_train"]])
        val_data = pd.concat(paquets[nb_packets_split_data["nb_packets_train"] :nb_packets_split_data["nb_packets_train"]+nb_packets_split_data["nb_packets_validation"]])
        test_data = pd.concat(paquets[nb_packets_split_data["nb_packets_train"]+nb_packets_split_data["nb_packets_validation"]: ])
        pbar.update(1)
        
        #store pre-processed data 
        train_data.to_csv(DATA_PATHS["processed"]["train"], index=False)
        val_data.to_csv(DATA_PATHS["processed"]["validation"], index=False)  
        test_data.to_csv(DATA_PATHS["processed"]["test"], index=False) 
        data.to_csv(DATA_PATHS["processed"]["all_data"], index=False)   
        pbar.update(1)
        

if __name__ == "__main__":
    run_preprocessing_pipeline()