# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Andrei Sasinovich
# Created Date: 28/08/21
# version ='1.0'
# ---------------------------------------------------------------------------
""" Test scripts for churn_library.py """

# pylint: disable=logging-fstring-interpolation,broad-except

import os
import logging
import pytest

import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_script_testing.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
)

TEST_DATA_FILE = "./data/bank_data.csv"


@pytest.fixture
def input_df():
    '''
    loading data fixture
    '''
    try:
        dataframe = cl.import_data(TEST_DATA_FILE)
        logging.info("Testing import_data: SUCCESS")
        return dataframe
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err


@pytest.fixture
def category_lst():
    '''
    categories fixture
    '''
    categories = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]

    return categories


def test_import():
    '''
    test data import
    '''
    try:
        dataframe = cl.import_data(TEST_DATA_FILE)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(input_df):
    '''
    test perform eda function
    '''
    try:
        cl.perform_eda(input_df)

        assert os.path.isfile(os.path.join(cl.EDA_SAVE_PATH, 'total_transaction_distrubution.png'))
        assert os.path.isfile(os.path.join(cl.EDA_SAVE_PATH, 'churn_distrubution.png'))
        assert os.path.isfile(os.path.join(cl.EDA_SAVE_PATH, 'customer_age_distrubution.png'))
        assert os.path.isfile(os.path.join(cl.EDA_SAVE_PATH, 'marital_status_distrubution.png'))
        assert os.path.isfile(os.path.join(cl.EDA_SAVE_PATH, 'heatmap.png'))

        logging.info("Testing perform eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform eda:: EDA image not found")
        raise err


def test_encoder_helper(input_df, category_lst):
    '''
    test encoder helper
    '''
    try:

        df_input_shape = input_df.shape
        dataframe = cl.encoder_helper(input_df, category_lst)

    except Exception as err:
        logging.error(f"Testing encoder_helper error: {err}")
        raise err

    try:
        assert dataframe.shape[0] == df_input_shape[0]
        assert dataframe.shape[1] == df_input_shape[1] + len(category_lst)
        logging.info("Testing encoder_helper: SUCCESS")

    except AssertionError as err:
        logging.error("Testing encoder_helper: Encoding error")
        raise err


def test_perform_feature_engineering(input_df, category_lst):
    '''
    test perform_feature_engineering
    '''
    try:

        dataframe = cl.encoder_helper(input_df, category_lst)
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(dataframe)

    except Exception as err:
        logging.error(f"Testing perform_feature_engineering: {err}")
        raise err

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0

        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0

        assert y_train.shape[0] > 0

        assert y_test.shape[0] > 0

        logging.info("Testing feature_engineering: SUCCESS")

    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: output dataframe shape error")
        raise err


def test_train_models(input_df, category_lst):
    '''
    test train_models
    '''
    try:

        dataframe = cl.encoder_helper(input_df, category_lst)
        df_100 = dataframe.iloc[:250, :]

        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(df_100)
        rf_model, lr_mode = cl.train_models(x_train, x_test, y_train, y_test)

    except Exception as err:
        logging.error(f"Testing train_models error: {err}")
        raise err

    try:
        assert rf_model is not None
        assert lr_mode is not None

        logging.info("Testing train_models: SUCCESS")

    except AssertionError as err:
        logging.error("Testing train_models: ERROR")
        raise err


if __name__ == "__main__":
    pass
