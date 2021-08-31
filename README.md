# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This main goal of the project is to turn data science code from Jupiter notebook into production-ready code.

The churn_library.py containing Python functions to train models and predict credit card customers that are most likely to churn. 


# How To Use

## Requirements

Install dependencies and libraries

```pip install -r requirements.txt```
 
Or install the following dependencies for this project:


- matplotlib
- pandas
- scikit-learn
- joblib
- seaborn


## Set output path

Output path for plots and trained models (churn_library.py):


- EDA_SAVE_PATH: path to save 

- RESULTS_SAVE_PATH: path to result plots

- MODELS_PATH: path to save models


## Input data
Example dataset: './data/bank_data.csv'

```
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]

    data = import_data('./data/bank_data.csv' )
    perform_eda(data)

    df_encoded = encoder_helper(data, category_lst)
    x_training, x_testing, y_training, y_testing = perform_feature_engineering(df_encoded)

```

## Run training

Run model training pipeline with the following command:

```python churn_library.py --data data/bank_data.csv```

The input data will be preprocessed and data charts will be saved in './images/eda' folder

Random Forest and Logistic Regression models will be trained. The models will be saved in './models' folder
The training results will be saved in './images/results' folder

OR use the following code for model training:

```
    rfc_model, lrc_model = train_models(x_training, x_testing, y_training, y_testing)
    save_models(rfc_model, lrc_model)

```





## Run prediction
```
    # Load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lrc_model = joblib.load('./models/logistic_model.pkl')

    # Test predict methods
    y_preds_rf = random_forest_predict(rfc_model, x_testing)
    y_preds_lr = logistic_regression_predict(lrc_model, x_testing)
```

## Unit testing ML code with Pytest
Run tests:

```pytest churn_script_logging_and_tests.py```


