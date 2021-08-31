# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Andrei Sasinovich
# Created Date: 28/08/21
# version ='1.0'
# ---------------------------------------------------------------------------
""" Predict Customer Churn: model training and prediction functions """

# pylint: disable=logging-fstring-interpolation,broad-except

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report


import joblib
import seaborn as sns
import pandas as pd

sns.set()

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
)

EDA_SAVE_PATH = './images/eda'
RESULTS_SAVE_PATH = './images/results'
MODELS_PATH = './models'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df_out: pandas dataframe
    '''
    try:
        dataframe = pd.read_csv(pth)
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        logging.info("Import data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Error: import_data file not found")
        raise err
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''

    try:

        plt.figure(figsize=(20, 10))

        sns.distplot(dataframe['Total_Trans_Ct'])
        plt.savefig(os.path.join(EDA_SAVE_PATH, 'total_transaction_distrubution.png'))

        plt.figure(figsize=(20, 10))
        dataframe['Churn'].hist()
        plt.savefig(os.path.join(EDA_SAVE_PATH, 'churn_distrubution.png'))

        plt.figure(figsize=(20, 10))
        dataframe['Customer_Age'].hist()
        plt.savefig(os.path.join(EDA_SAVE_PATH, 'customer_age_distrubution.png'))

        plt.figure(figsize=(20, 10))
        dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(os.path.join(EDA_SAVE_PATH, 'marital_status_distrubution.png'))

        plt.figure(figsize=(20, 12))
        sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(EDA_SAVE_PATH, 'heatmap.png'))
        logging.info('All EDA images have been saved')

    except FileNotFoundError:
        logging.error('Error: error occurred saving image, wrong file path')


def encoder_helper(dataframe, category_list):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_list: list of columns that contain categorical features

    output:
            dataframe: pandas dataframe with new columns for
    '''

    try:
        for category in category_list:
            lst = []
            groups = dataframe.groupby(category).mean()['Churn']

            for val in dataframe[category]:
                lst.append(groups.loc[val])

            column_name = category + '_Churn'
            dataframe[column_name] = lst

        return dataframe
    except KeyError as err:
        logging.error(f'Error: Category {err} not found')


def perform_feature_engineering(dataframe):
    '''
    input:
              dataframe: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn',
    ]
    x_data = pd.DataFrame()

    try:
        y_data = dataframe['Churn']
        x_data[keep_cols] = dataframe[keep_cols]

        # train test split
        x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=42
        )
        return x_data_train, x_data_test, y_data_train, y_data_test

    except Exception as err:
        logging.error(f'Error: error occurred in feature_engineering {err}')


def classification_report_image(
    y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Save Random Forest result image
    plt.figure(figsize=(8, 8))
    # plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {'fontsize': 10},
        fontproperties='monospace',
    )
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {'fontsize': 10},
        fontproperties='monospace',
    )
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_SAVE_PATH, 'logistic_results.png'))

    # Save Logistic Regression result image
    plt.figure(figsize=(8, 8))
    plt.text(
        0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties='monospace'
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {'fontsize': 10},
        fontproperties='monospace',
    )
    plt.text(
        0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties='monospace'
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {'fontsize': 10},
        fontproperties='monospace',
    )
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_SAVE_PATH, 'rf_results.png'))


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 8))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(os.path.join(output_pth, 'feature_importance.png'))
        logging.info('Feature_ importance figure has been saved')

    except Exception as err:
        logging.error(f'Error: feature_importance {err}')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    try:
        logging.info('Starting model training')
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy'],
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        # do not write images if train data is small, used for test
        if x_train.shape[0] > 1000:
            feature_importance_plot(cv_rfc, x_train, './images/results')

            # scores
            classification_report_image(
                y_train,
                y_test,
                y_train_preds_lr,
                y_train_preds_rf,
                y_test_preds_lr,
                y_test_preds_rf,
            )

        logging.info('Training has been finished successfully')
        return cv_rfc, lrc

    except Exception as err:
        logging.error(f'Error: model training error {err}')
        return None, None


def save_models(cv_rfc, lrc, rf_name='rfc_model.pkl', lr_name='logistic_model.pkl'):
    '''
    Store models
    input:
              cv_rfc: Random Forest model
              lrc: Logistic Regression model
              rf_name: Random Forest model file name
              lr_name: Logistic Regression model file name
    output:
              None
    '''
    # save best model
    try:
        joblib.dump(cv_rfc.best_estimator_, os.path.join(MODELS_PATH, rf_name))
        joblib.dump(lrc, os.path.join(MODELS_PATH, lr_name))
        logging.info('Models have been saved successfully')
    except Exception as err:
        logging.error(f'Error: model saving error {err}')
        return


def roc_curve_compare_and_plot(rf_model, lr_model, x_test, y_test):
    '''
    Plot and save roc_curve
    input:
              rf_model: Random Forest model
              lr_model: Logistic Regression model
              x_test: X data
              y_test: Y data
    output:
              None
    '''
    # store roc curve result
    try:

        lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
        plt.figure(figsize=(15, 8))
        axis = plt.gca()
        rf_disp = plot_roc_curve(rf_model, x_test, y_test, ax=axis, alpha=0.8)
        lrc_plot.plot(ax=axis, alpha=0.8)
        plt.savefig(os.path.join(RESULTS_SAVE_PATH, 'roc_curve_result.png'))
    except FileNotFoundError:
        logging.error('Error: roc_curve_result saving error')
        return


def random_forest_predict(rf_model, x_data):
    '''
    Random forest predict function
    input:
              rf_model: Random forest model
              x_data: X testing data
    output:
              y_predictions_rf: predictions
    '''
    y_predictions_rf = rf_model.predict(x_data)
    return y_predictions_rf


def logistic_regression_predict(lr_model, x_data):
    '''
    Logistic Regression predict function
    input:
              rfc_model: Logistic Regression model
              x_data: X testing data
    output:
              y_predictions_rlr: predictions
    '''
    y_predictions_lr = lr_model.predict(x_data)
    return y_predictions_lr


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="path to input data.csv file")
    args = vars(ap.parse_args())

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]

    data = import_data(args["data"])
    perform_eda(data)

    df_encoded = encoder_helper(data, category_lst)
    x_training, x_testing, y_training, y_testing = perform_feature_engineering(df_encoded)
    rfc_model, lrc_model = train_models(x_training, x_testing, y_training, y_testing)
    save_models(rfc_model, lrc_model)

    # Load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lrc_model = joblib.load('./models/logistic_model.pkl')

    # Plot roc curves
    roc_curve_compare_and_plot(rfc_model, lrc_model, x_testing, y_testing)

    # Test predict methods
    y_preds_rf = random_forest_predict(rfc_model, x_testing)
    y_preds_lr = logistic_regression_predict(lrc_model, x_testing)
