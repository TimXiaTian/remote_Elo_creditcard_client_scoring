import numpy as np
import pandas as pd


def get_columns_to_delete(train, conditions):
    """
    Identify columns to delete based on specified conditions.

    Parameters:
    train (pd.DataFrame): The training dataset.
    conditions (list of tuples): A list of conditions where each tuple contains substrings to be checked in column names.

    Returns:
    list: A list of columns that match any of the specified conditions.
    """
    del_cols = []
    for col in train.columns:
        for condition in conditions:
            if all(sub in col for sub in condition):
                del_cols.append(col)
                break
    return del_cols


def load_and_process_data(train_path, test_path, inf_cols):
    """
    Load train and test datasets, replace infinite values in specified columns, and filter out outliers.

    Parameters:
    train_path (str): Path to the training dataset CSV file.
    test_path (str): Path to the testing dataset CSV file.
    inf_cols (list of str): List of columns to replace infinite values.

    Returns:
    pd.DataFrame: Processed training dataset.
    pd.DataFrame: Processed testing dataset.
    pd.DataFrame: Training dataset without outliers.
    np.array: Target values from the training dataset.
    np.array: Target values from the training dataset without outliers.
    np.array: Binary target values indicating outliers.
    """
    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Replace infinite values
    for df in [train, test]:
        df[inf_cols] = df[inf_cols].replace(np.inf, df[inf_cols].replace(np.inf, -99).max().max())

    # Filter out outliers
    ntrain = train[train['outliers'] == 0]

    # Prepare target variables
    target = train['target'].values
    ntarget = ntrain['target'].values
    target_binary = train['outliers'].values

    print('train:', train.shape)
    print('ntrain:', ntrain.shape)

    return train, test, ntrain, target, ntarget, target_binary
