import pandas as pd
import numpy as np
import datetime


# Control the memory usage of a DataFrame
def reduce_mem_usage(df, verbose=True):
    """
    Reduce the memory usage of a DataFrame by changing the data type of its columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to reduce memory usage.
    verbose (bool): Whether to print memory usage information.

    Returns:
    df (pd.DataFrame): DataFrame with reduced memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return df


def change_object_cols(series):
    """
    convert the values of categorical columns to their corresponding integer encoding.
    param series
    return: Integer numpy array
    """

    value = series.unique().tolist()
    value.sort()
    return series.map(pd.Series(range(len(value)), index=value)).values


def merge_first_act_month(df_, train, test):
    """
    add the first activity month column to the dataframe

    param df_: dataframe we want to add the column
    param train: train data which has the first activity month column and card_id column
    param test: test data which has the first activity month column and card_id column

    return:
    """
    train_test = pd.concat([train[['card_id', 'first_active_month']],
                            test[['card_id', 'first_active_month']]],
                           axis=0,
                           ignore_index=True)
    df_ = (df_
           .merge(train_test[['card_id', 'first_active_month']],
                  on=['card_id'],
                  how='left'))

    return df_


def gen_date_features(df):
    """
    This function preprocesses the datasets by extracting and transforming date-related features.
    Preprocess the datasets to extract year from 'first_active_month',
    Calculate elapsed time in days from 'first_active_month' to a reference date,
    Extract week of the year, day of the year, and month from 'first_active_month'.

    Parameters:
    df (pd.DataFrame): The dataset containing 'first_active_month' columns.

    Returns:
    df (pd.DataFrame): df with new generated 'year', 'elapsed_time', 'weekofyear', 'dayofyear', 'month' columns.
    """

    df['year'] = df['first_active_month'].fillna('0-0').apply(lambda x: int(str(x).split('-')[0]))
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 3, 1) - df['first_active_month'].dt.date).dt.days
    df['weekofyear'] = df['first_active_month'].dt.isocalendar().week
    df['dayofyear'] = df['first_active_month'].dt.dayofyear
    df['month'] = df['first_active_month'].dt.month

    return df


# Convert days to months
def month_trans(x):
    """
    Convert days to months.

    Parameters:
    x (int): Number of days.

    Returns:
    int: Number of months.
    """
    return x // 30


# Convert days to weeks
def week_trans(x):
    """
    Convert days to weeks.

    Parameters:
    x (int): Number of days.

    Returns:
    int: Number of weeks.
    """
    return x // 7


# Preprocess transaction data
def get_expand_common(df_):
    """
    Preprocess transaction data by filling missing values, creating new features, and reducing memory usage.

    Parameters:
    df_ (pd.DataFrame): The transaction data.

    Returns:
    df (pd.DataFrame): Preprocessed transaction data.
    """

    df = df_.copy()

    # Missing value annotation
    df['category_2'].fillna(1.0, inplace=True)
    df['category_3'].fillna('A', inplace=True)
    df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
    df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    df['installments'].replace(-1, np.nan, inplace=True)
    df['installments'].replace(999, np.nan, inplace=True)
    df['installments'].replace(0, 1, inplace=True)

    df['purchase_amount'] = np.round(df['purchase_amount'] / 0.00150265118 + 497.06, 8)
    df['purchase_amount'] = df.purchase_amount.apply(lambda x: np.round(x))

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['purchase_hour'] = df['purchase_date'].dt.hour
    df['year'] = df['purchase_date'].dt.year
    df['month'] = df['purchase_date'].dt.month
    df['day'] = df['purchase_date'].dt.day
    df['hour'] = df['purchase_date'].dt.hour
    df['weekofyear'] = df['purchase_date'].dt.isocalendar().week
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
    df = df.sort_values(['card_id', 'purchase_date'])
    df['purchase_date_floorday'] = df['purchase_date'].dt.floor('d')  # Remove time smaller than day

    # Relative time from activation time
    df['purchase_day_since_active_day'] = df['purchase_date_floorday'] - df['first_active_month']
    df['purchase_day_since_active_day'] = df['purchase_day_since_active_day'].dt.days
    df['purchase_month_since_active_day'] = df['purchase_day_since_active_day'].agg(month_trans).values
    df['purchase_week_since_active_day'] = df['purchase_day_since_active_day'].agg(week_trans).values

    # Relative time from the last day
    ht_card_id_gp = df.groupby('card_id')
    df['purchase_day_since_reference_day'] = ht_card_id_gp['purchase_date_floorday'].transform('max') - df[
        'purchase_date_floorday']
    df['purchase_day_since_reference_day'] = df['purchase_day_since_reference_day'].dt.days
    df['purchase_week_since_reference_day'] = df['purchase_day_since_reference_day'].agg(week_trans).values
    df['purchase_month_since_reference_day'] = df['purchase_day_since_reference_day'].agg(month_trans).values

    df['purchase_day_diff'] = df['purchase_date_floorday'].shift()
    df['purchase_day_diff'] = df['purchase_date_floorday'].values - df['purchase_day_diff'].values
    df['purchase_day_diff'] = df['purchase_day_diff'].dt.days
    df['purchase_week_diff'] = df['purchase_day_diff'].agg(week_trans).values
    df['purchase_month_diff'] = df['purchase_day_diff'].agg(month_trans).values

    decay_rates = {
        'ddgd_98': 0.98,
        'ddgd_99': 0.99,
        'wdgd_96': 0.96,
        'wdgd_97': 0.97,
        'mdgd_90': 0.90,
        'mdgd_80': 0.80
    }
    for key, rate in decay_rates.items():
        df[f'purchase_amount_{key}'] = df['purchase_amount'] * df['purchase_day_since_reference_day'].apply(
            lambda x: rate ** x).values

    df = reduce_mem_usage(df)

    return df
