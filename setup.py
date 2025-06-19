from setuptools import setup, find_packages

setup(
    name='saud_detector',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            # Data preprocessing
            'run-preprocessing=src.features.preprocessing:run_preprocessing_pipeline',
            # Model training 
            'train_model=src.models.train_model:main',
            # Model prediction 
            'predict_model=src.models.predict_model:main',
            # model evaluation classification performances 
            'eval_classification_perfs = src.models.evaluate_classification_perfs:main', 
            # saud threshold search 
            'search_best_threshold = src.models.threshold_search:main' 
            # get the identified saud 
            'get_sauds = src.models.get_identified_saud_stats:main'
        ],
    },
)
