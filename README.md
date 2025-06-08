
```
saud_detector
├─ LICENSE
├─ README_finale.md
├─ config_template.py
├─ logs
│  ├─ avaluate_logs
│  ├─ prediction_logs
│  ├─ search_threshold_logs
│  ├─ training_logs
│  └─ uncertainty_estimation_logs
├─ models
│  ├─ best_MLP.pkl
│  ├─ best_oneClassSVM.pkl
│  └─ best_xgboost.pkl
├─ reports
│  ├─ AUD_nonAUD_classif
│  │  └─ conf_matrices
│  │     ├─ MLP.csv
│  │     ├─ oneClassSVM.csv
│  │     └─ xgboost.csv
│  ├─ identified_saud
│  └─ sAUD_threshold_search
│     ├─ MLP
│     │  ├─ plot_hcc_age_onset_by_seuils.pdf
│     │  └─ wasserstein_distance_by_threshold.csv
│     ├─ oneClassSVM
│     │  ├─ plot_hcc_age_onset_by_seuils.pdf
│     │  └─ wasserstein_distance_by_threshold.csv
│     └─ xgboost
│        ├─ plot_hcc_age_onset_by_seuils.pdf
│        └─ wasserstein_distance_by_threshold.csv
├─ requirement.txt
├─ setup.py
└─ src
   ├─ __init__.py
   ├─ features
   │  ├─ __init__.py
   │  └─ preprocessing.py
   └─ models
      ├─ MLP.py
      ├─ __init_.py
      ├─ available_models.py
      ├─ base.py
      ├─ evaluate_classification_perfs.py
      ├─ oneClassSVM.py
      ├─ predict_model.py
      ├─ threshold_search.py
      ├─ train_model.py
      ├─ uncertainty_estimation.py
      ├─ utils.py
      ├─ xgb_config.json
      └─ xgboost.py

```
