import gc
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm_notebook
from data_processing import reduce_mem_usage


# construct basic feature
def aggregate_transactions(df_, prefix):
    """
    Aggregate transaction data to create features.

    Purpose:
    This function aggregates transaction data to generate statistical features for each card_id.

    Parameters:
    df_ (pd.DataFrame): The transaction data.
    prefix (str): Prefix for the column names of the aggregated features.

    Returns:
    agg_df (pd.DataFrame): Aggregated features for each card_id.
    """
    df = df_.copy()

    df['month_diff'] = (datetime.datetime.today() - df['purchase_date']).dt.days // 30
    df['month_diff'] = df['month_diff'].astype(int)
    df['month_diff'] += df['month_lag']

    df['price'] = df['purchase_amount'] / df['installments']
    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

    df.loc[:, 'purchase_date'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9

    agg_func = {
        'category_1': ['mean'],
        'category_2': ['mean'],
        'category_3': ['mean'],
        'installments': ['mean', 'max', 'min', 'std'],
        'month_lag': ['nunique', 'mean', 'max', 'min', 'std'],
        'month': ['nunique', 'mean', 'max', 'min', 'std'],
        'hour': ['nunique', 'mean', 'max', 'min', 'std'],
        'weekofyear': ['nunique', 'mean', 'max', 'min', 'std'],
        'dayofweek': ['nunique', 'mean'],
        'weekend': ['mean'],
        'year': ['nunique'],
        'card_id': ['size', 'count'],
        'purchase_date': ['max', 'min'],
        'price': ['mean', 'max', 'min', 'std'],
        'duration': ['mean', 'min', 'max', 'std', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'std', 'skew'],
    }

    for col in ['category_2', 'category_3']:
        df[col + '_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        agg_func[col + '_mean'] = ['mean']

    agg_df = df.groupby(['card_id']).agg(agg_func)
    agg_df.columns = [prefix + '_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)

    return agg_df


# derive features on our basic features
def get_quantile(x, percentiles=(0.1, 0.25, 0.75, 0.9)):
    """
    Calculate quantiles for the given data.

    Parameters:
    x (np.array): The data array.
    percentiles (tuple): The quantiles to calculate.

    Returns:
    sts_feas (list): List of quantile values.
    """
    x_len = len(x)
    x = np.sort(x)
    sts_feas = []
    for per_ in percentiles:
        if per_ == 1:
            sts_feas.append(x[x_len - 1])
        else:
            sts_feas.append(x[int(x_len * per_)])
    return sts_feas


def get_cardf_tran(df_, month=3, prefix='_'):
    """
    Generate card transaction features.

    Parameters:
    df_ (pd.DataFrame): The transaction data.
    month (int): Reference month for calculating certain features.
    prefix (str): Prefix for the column names of the generated features.

    Returns:
    cardid_features (pd.DataFrame): Generated features for each card_id.
    """
    df = df_.copy()
    if prefix == 'hist_cardf_':
        df['month_to_now'] = (datetime.date(2018, month, 1) - df['purchase_date_floorday'].dt.date).dt.days

    df['month_diff'] = (datetime.datetime.today() - df['purchase_date']).dt.days // 30
    df['month_diff'] = df['month_diff'].astype(int)
    df['month_diff'] += df['month_lag']

    print('*' * 30, 'Part1, whole data', '*' * 30)
    cardid_features = pd.DataFrame()
    cardid_features['card_id'] = df['card_id'].unique()
    print('*' * 30, 'Traditional Features', '*' * 30)
    ht_card_id_gp = df.groupby('card_id')
    cardid_features['card_id_cnt'] = ht_card_id_gp['authorized_flag'].count().values

    if prefix == 'hist_cardf_':
        cardid_features['card_id_isau_mean'] = ht_card_id_gp['authorized_flag'].mean().values
        cardid_features['card_id_isau_sum'] = ht_card_id_gp['authorized_flag'].sum().values

    cardid_features['month_diff_mean'] = ht_card_id_gp['month_diff'].mean().values
    cardid_features['month_diff_median'] = ht_card_id_gp['month_diff'].median().values

    if prefix == 'hist_cardf_':
        cardid_features['reference_day'] = ht_card_id_gp['purchase_date_floorday'].max().values
        cardid_features['first_day'] = ht_card_id_gp['purchase_date_floorday'].min().values
        cardid_features['activation_day'] = ht_card_id_gp['first_active_month'].max().values

        # first to activation day
        cardid_features['first_to_activation_day'] = (
                cardid_features['first_day'] - cardid_features['activation_day']).dt.days
        # activation to reference day
        cardid_features['activation_to_reference_day'] = (
                cardid_features['reference_day'] - cardid_features['activation_day']).dt.days
        # first to last day
        cardid_features['first_to_reference_day'] = (
                cardid_features['reference_day'] - cardid_features['first_day']).dt.days
        # reference day to now
        cardid_features['reference_day_to_now'] = (
                datetime.date(2018, month, 1) - cardid_features['reference_day'].dt.date).dt.days
        # first day to now
        cardid_features['first_day_to_now'] = (
                datetime.date(2018, month, 1) - cardid_features['first_day'].dt.date).dt.days

        print('card_id(month_lag, min to reference day):min')
        cardid_features['card_id_month_lag_min'] = ht_card_id_gp['month_lag'].agg('min').values
        # is_purchase_before_activation,first_to_reference_day_divide_activation_to_reference_day
        cardid_features['is_purchase_before_activation'] = cardid_features['first_to_activation_day'] < 0
        cardid_features['is_purchase_before_activation'] = cardid_features['is_purchase_before_activation'].astype(int)
        cardid_features['first_to_reference_day_divide_activation_to_reference_day'] = cardid_features[
                                                                                           'first_to_reference_day'] / (
                                                                                               cardid_features[
                                                                                                   'activation_to_reference_day'] + 0.01)
        cardid_features['days_per_count'] = cardid_features['first_to_reference_day'].values / cardid_features[
            'card_id_cnt'].values

    if prefix == 'new_cardf_':
        print(' Eight time features, ')
        cardid_features['reference_day'] = ht_card_id_gp['reference_day'].last().values
        cardid_features['first_day'] = ht_card_id_gp['purchase_date_floorday'].min().values
        cardid_features['last_day'] = ht_card_id_gp['purchase_date_floorday'].max().values
        cardid_features['activation_day'] = ht_card_id_gp['first_active_month'].max().values
        # reference to first day
        cardid_features['reference_day_to_first_day'] = (
                cardid_features['first_day'] - cardid_features['reference_day']).dt.days
        # reference to last day
        cardid_features['reference_day_to_last_day'] = (
                cardid_features['last_day'] - cardid_features['reference_day']).dt.days
        # first to last day
        cardid_features['first_to_last_day'] = (cardid_features['last_day'] - cardid_features['first_day']).dt.days
        # activation to first day
        cardid_features['activation_to_first_day'] = (
                cardid_features['first_day'] - cardid_features['activation_day']).dt.days
        # activation to first day
        cardid_features['activation_to_last_day'] = (
                cardid_features['last_day'] - cardid_features['activation_day']).dt.days
        # last day to now
        cardid_features['reference_day_to_now'] = (
                datetime.date(2018, month, 1) - cardid_features['reference_day'].dt.date).dt.days
        # first day to now
        cardid_features['first_day_to_now'] = (
                datetime.date(2018, month, 1) - cardid_features['first_day'].dt.date).dt.days

        print('card_id(month_lag, min to reference day):min')
        cardid_features['card_id_month_lag_max'] = ht_card_id_gp['month_lag'].agg('max').values
        cardid_features['first_to_last_day_divide_reference_to_last_day'] = cardid_features['first_to_last_day'] / (
                cardid_features['reference_day_to_last_day'] + 0.01)
        cardid_features['days_per_count'] = cardid_features['first_to_last_day'].values / cardid_features[
            'card_id_cnt'].values

    for f in ['reference_day', 'first_day', 'last_day', 'activation_day']:
        try:
            del cardid_features[f]
        except:
            print(f, 'not exist！！！')

    print('card id(city_id,installments,merchant_category_id,.......):nunique, cnt/nunique')
    for col in tqdm_notebook(
            ['category_1', 'category_2', 'category_3', 'state_id', 'city_id', 'installments', 'merchant_id',
             'merchant_category_id', 'subsector_id', 'month_lag', 'purchase_date_floorday']):
        cardid_features['card_id_%s_nunique' % col] = ht_card_id_gp[col].nunique().values
        cardid_features['card_id_cnt_divide_%s_nunique' % col] = cardid_features['card_id_cnt'].values / \
                                                                 cardid_features['card_id_%s_nunique' % col].values

    print('card_id(purchase_amount & degrade version ):mean,sum,std,median,quantile(10,25,75,90)')
    for col in tqdm_notebook(['installments', 'purchase_amount', 'purchase_amount_ddgd_98', 'purchase_amount_ddgd_99',
                              'purchase_amount_wdgd_96', 'purchase_amount_wdgd_97', 'purchase_amount_mdgd_90',
                              'purchase_amount_mdgd_80']):
        if col == 'purchase_amount':
            for opt in ['sum', 'mean', 'std', 'median', 'max', 'min']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values

            cardid_features['card_id_' + col + '_range'] = cardid_features['card_id_' + col + '_max'].values - \
                                                           cardid_features['card_id_' + col + '_min'].values
            percentiles = ht_card_id_gp[col].apply(lambda x: get_quantile(x, percentiles=[0.025, 0.25, 0.75, 0.975]))

            cardid_features[col + '_2.5_quantile'] = percentiles.map(lambda x: x[0]).values
            cardid_features[col + '_25_quantile'] = percentiles.map(lambda x: x[1]).values
            cardid_features[col + '_75_quantile'] = percentiles.map(lambda x: x[2]).values
            cardid_features[col + '_97.5_quantile'] = percentiles.map(lambda x: x[3]).values
            cardid_features['card_id_' + col + '_range2'] = cardid_features[col + '_97.5_quantile'].values - \
                                                            cardid_features[col + '_2.5_quantile'].values
            del cardid_features[col + '_2.5_quantile'], cardid_features[col + '_97.5_quantile']
            gc.collect()
        else:
            for opt in ['sum']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values

    print('*' * 30, 'Pivot Features', '*' * 30)
    print(
        'Count  Pivot')
    for pivot_col in tqdm_notebook(
            ['category_1', 'category_2', 'category_3', 'month_lag', 'subsector_id', 'weekend']):  # 'city_id',,

        tmp = df.groupby(['card_id', pivot_col])['merchant_id'].count().to_frame(pivot_col + '_count')
        tmp.reset_index(inplace=True)

        tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col, values=pivot_col + '_count',
                                   fill_value=0)
        tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_cnt_pivot_' + str(col) for col in tmp_pivot.columns]
        tmp_pivot.reset_index(inplace=True)
        cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

        if pivot_col != 'weekend' and pivot_col != 'installments':
            tmp = df.groupby(['card_id', pivot_col])['purchase_date_floorday'].nunique().to_frame(
                pivot_col + '_purchase_date_floorday_nunique')
            tmp1 = df.groupby(['card_id'])['purchase_date_floorday'].nunique().to_frame(
                'purchase_date_floorday_nunique')
            tmp.reset_index(inplace=True)
            tmp1.reset_index(inplace=True)
            tmp = tmp.merge(tmp1, on='card_id', how='left')
            tmp[pivot_col + '_day_nunique_pct'] = tmp[pivot_col + '_purchase_date_floorday_nunique'].values / tmp[
                'purchase_date_floorday_nunique'].values

            tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col,
                                       values=pivot_col + '_day_nunique_pct', fill_value=0)
            tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_day_nunique_pct_' + str(col) for col in
                                 tmp_pivot.columns]
            tmp_pivot.reset_index(inplace=True)
            cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

    if prefix == 'new_cardf_':
        ######## record of spending before the card is activated  ##############
        print('*' * 30, 'Part2， data with time less than activation day', '*' * 30)
        df_part = df.loc[df.purchase_date < df.first_active_month]

        cardid_features_part = pd.DataFrame()
        cardid_features_part['card_id'] = df_part['card_id'].unique()
        ht_card_id_part_gp = df_part.groupby('card_id')
        cardid_features_part['card_id_part_cnt'] = ht_card_id_part_gp['authorized_flag'].count().values

        print('card_id(purchase_amount): sum')
        for col in tqdm_notebook(['purchase_amount']):
            for opt in ['sum', 'mean']:
                cardid_features_part['card_id_part_' + col + '_' + opt] = ht_card_id_part_gp[col].agg(opt).values

        cardid_features = cardid_features.merge(cardid_features_part, on='card_id', how='left')
        cardid_features['card_id_part_purchase_amount_sum_percent'] = cardid_features[
                                                                          'card_id_part_purchase_amount_sum'] / (
                                                                              cardid_features[
                                                                                  'card_id_purchase_amount_sum'] + 0.01)

    cardid_features = reduce_mem_usage(cardid_features)

    new_col_names = []
    for col in cardid_features.columns:
        if col == 'card_id':
            new_col_names.append(col)
        else:
            new_col_names.append(prefix + col)
    cardid_features.columns = new_col_names

    return cardid_features


# We need to further consider the user behavior features of the past two months:
def get_cardf_tran_last2(df_, month=3, prefix='last2_'):
    """
    Generate card transaction features for the last two months.

    Purpose:
    This function generates various transaction features for each card_id for the last two months of transactions.

    Parameters:
    df_ (pd.DataFrame): The transaction data.
    month (int): Reference month for calculating certain features.
    prefix (str): Prefix for the column names of the generated features.

    Returns:
    cardid_features (pd.DataFrame): Generated features for each card_id.
    """
    df = df_.loc[df_.month_lag >= -2].copy()
    print('*' * 30, 'Part1, whole data', '*' * 30)
    cardid_features = pd.DataFrame()
    cardid_features['card_id'] = df['card_id'].unique()

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
    df['month_diff'] = df['month_diff'].astype(int)
    df['month_diff'] += df['month_lag']

    print('*' * 30, 'Traditional Features', '*' * 30)
    ht_card_id_gp = df.groupby('card_id')
    print(' card id : count')
    cardid_features['card_id_cnt'] = ht_card_id_gp['authorized_flag'].count().values

    cardid_features['card_id_isau_mean'] = ht_card_id_gp['authorized_flag'].mean().values
    cardid_features['card_id_isau_sum'] = ht_card_id_gp['authorized_flag'].sum().values

    cardid_features['month_diff_mean'] = ht_card_id_gp['month_diff'].mean().values

    print('card id(city_id,installments,merchant_category_id,.......):nunique, cnt/nunique')
    for col in tqdm_notebook(
            ['state_id', 'city_id', 'installments', 'merchant_id', 'merchant_category_id', 'purchase_date_floorday']):
        cardid_features['card_id_%s_nunique' % col] = ht_card_id_gp[col].nunique().values
        cardid_features['card_id_cnt_divide_%s_nunique' % col] = cardid_features['card_id_cnt'].values / \
                                                                 cardid_features['card_id_%s_nunique' % col].values

    for col in tqdm_notebook(
            ['purchase_amount', 'purchase_amount_ddgd_98', 'purchase_amount_wdgd_96', 'purchase_amount_mdgd_90',
             'purchase_amount_mdgd_80']):  # ,'purchase_amount_ddgd_98','purchase_amount_ddgd_99','purchase_amount_wdgd_96','purchase_amount_wdgd_97','purchase_amount_mdgd_90','purchase_amount_mdgd_80']):
        if col == 'purchase_amount':
            for opt in ['sum', 'mean', 'std', 'median']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values
        else:
            for opt in ['sum']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values

    print('*' * 30, 'Pivot Features', '*' * 30)
    print(
        'Count  Pivot')

    for pivot_col in tqdm_notebook(
            ['category_1', 'category_2', 'category_3', 'month_lag', 'subsector_id', 'weekend']):  # 'city_id',

        tmp = df.groupby(['card_id', pivot_col])['merchant_id'].count().to_frame(pivot_col + '_count')
        tmp.reset_index(inplace=True)

        tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col, values=pivot_col + '_count',
                                   fill_value=0)
        tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_cnt_pivot_' + str(col) for col in tmp_pivot.columns]
        tmp_pivot.reset_index(inplace=True)
        cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

        if pivot_col != 'weekend' and pivot_col != 'installments':
            tmp = df.groupby(['card_id', pivot_col])['purchase_date_floorday'].nunique().to_frame(
                pivot_col + '_purchase_date_floorday_nunique')
            tmp1 = df.groupby(['card_id'])['purchase_date_floorday'].nunique().to_frame(
                'purchase_date_floorday_nunique')
            tmp.reset_index(inplace=True)
            tmp1.reset_index(inplace=True)
            tmp = tmp.merge(tmp1, on='card_id', how='left')
            tmp[pivot_col + '_day_nunique_pct'] = tmp[pivot_col + '_purchase_date_floorday_nunique'].values / tmp[
                'purchase_date_floorday_nunique'].values

            tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col,
                                       values=pivot_col + '_day_nunique_pct', fill_value=0)
            tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_day_nunique_pct_' + str(col) for col in
                                 tmp_pivot.columns]
            tmp_pivot.reset_index(inplace=True)
            cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

    cardid_features = reduce_mem_usage(cardid_features)

    new_col_names = []
    for col in cardid_features.columns:
        if col == 'card_id':
            new_col_names.append(col)
        else:
            new_col_names.append(prefix + col)
    cardid_features.columns = new_col_names

    return cardid_features


# Further second-order cross-feature derivation
def successive_aggregates(df_, prefix='levelAB_'):
    """
    Generate second-order cross-features.

    Parameters:
    df_ (pd.DataFrame): The transaction data.
    prefix (str): Prefix for the column names of the generated features.

    Returns:
    cardid_features (pd.DataFrame): Generated second-order cross-features for each card_id.
    """
    df = df_.copy()
    cardid_features = pd.DataFrame()
    cardid_features['card_id'] = df['card_id'].unique()

    level12_nunique = [('month_lag', 'state_id'), ('month_lag', 'city_id'), ('month_lag', 'subsector_id'),
                       ('month_lag', 'merchant_category_id'), ('month_lag', 'merchant_id'),
                       ('month_lag', 'purchase_date_floorday'), \
                       ('subsector_id', 'merchant_category_id'), ('subsector_id', 'merchant_id'),
                       ('subsector_id', 'purchase_date_floorday'), ('subsector_id', 'month_lag'), \
                       ('merchant_category_id', 'merchant_id'), ('merchant_category_id', 'purchase_date_floorday'),
                       ('merchant_category_id', 'month_lag'), \
                       ('purchase_date_floorday', 'merchant_id'), ('purchase_date_floorday', 'merchant_category_id'),
                       ('purchase_date_floorday', 'subsector_id')]
    for col_level1, col_level2 in tqdm_notebook(level12_nunique):
        level1 = df.groupby(['card_id', col_level1])[col_level2].nunique().to_frame(col_level2 + '_nunique')
        level1.reset_index(inplace=True)

        level2 = level1.groupby('card_id')[col_level2 + '_nunique'].agg(['mean', 'max', 'std'])
        level2 = pd.DataFrame(level2)
        level2.columns = [col_level1 + '_' + col_level2 + '_nunique_' + col for col in level2.columns.values]
        level2.reset_index(inplace=True)

        cardid_features = cardid_features.merge(level2, on='card_id', how='left')

    level12_count = ['month_lag', 'state_id', 'city_id', 'subsector_id', 'merchant_category_id', 'merchant_id',
                     'purchase_date_floorday']
    for col_level in tqdm_notebook(level12_count):
        level1 = df.groupby(['card_id', col_level])['merchant_id'].count().to_frame(col_level + '_count')
        level1.reset_index(inplace=True)

        level2 = level1.groupby('card_id')[col_level + '_count'].agg(['mean', 'max', 'std'])
        level2 = pd.DataFrame(level2)
        level2.columns = [col_level + '_count_' + col for col in level2.columns.values]
        level2.reset_index(inplace=True)

        cardid_features = cardid_features.merge(level2, on='card_id', how='left')

    level12_meansum = [('month_lag', 'purchase_amount'), ('state_id', 'purchase_amount'),
                       ('city_id', 'purchase_amount'), ('subsector_id', 'purchase_amount'), \
                       ('merchant_category_id', 'purchase_amount'), ('merchant_id', 'purchase_amount'),
                       ('purchase_date_floorday', 'purchase_amount')]
    for col_level1, col_level2 in tqdm_notebook(level12_meansum):
        level1 = df.groupby(['card_id', col_level1])[col_level2].sum().to_frame(col_level2 + '_sum')
        level1.reset_index(inplace=True)

        level2 = level1.groupby('card_id')[col_level2 + '_sum'].agg(['mean', 'max', 'std'])
        level2 = pd.DataFrame(level2)
        level2.columns = [col_level1 + '_' + col_level2 + '_sum_' + col for col in level2.columns.values]
        level2.reset_index(inplace=True)

        cardid_features = cardid_features.merge(level2, on='card_id', how='left')

    cardid_features = reduce_mem_usage(cardid_features)

    new_col_names = []
    for col in cardid_features.columns:
        if col == 'card_id':
            new_col_names.append(col)
        else:
            new_col_names.append(prefix + col)
    cardid_features.columns = new_col_names

    return cardid_features


def merge_datasets(train, test, datasets, keys):
    '''
    Merges multiple datasets into the training and testing datasets based on specified keys.

    Parameters:
    train (pd.DataFrame): The training dataset to be merged with additional datasets.
    test (pd.DataFrame): The testing dataset to be merged with additional datasets.
    datasets (list of pd.DataFrame): A list of datasets to merge into the train and test datasets.
    keys (list of str): A list of keys on which the merge operations will be performed. Each key corresponds to a
                        / dataset  in the datasets list.

    Returns:
    pd.DataFrame: The updated training dataset after merging with the additional datasets.
    pd.DataFrame: The updated testing dataset after merging with the additional datasets.
    '''
    for dataset, key in zip(datasets, keys):
        train = pd.merge(train, dataset, on=key, how='left')
        test = pd.merge(test, dataset, on=key, how='left')
        print(train.shape)
        print(test.shape)
    return train, test


def create_outliers_feature(train, threshold=-30):
    """
    Create an 'outliers' feature based on the target column in the training dataset.

    Parameters:
    train (pd.DataFrame): The training dataset containing the 'target' column.
    threshold (float): The threshold below which a target value is considered an outlier.

    Returns:
    pd.DataFrame: The updated training dataset with the 'outliers' column.
    """
    train['outliers'] = 0
    train.loc[train['target'] < threshold, 'outliers'] = 1
    print(train['outliers'].value_counts())
    return train


def add_outliers_mean(train, test, features):
    """
    Add mean outliers features for specified features in both train and test datasets.

    Parameters:
    train (pd.DataFrame): The training dataset containing the 'outliers' and specified features.
    test (pd.DataFrame): The testing dataset containing the specified features.
    features (list of str): The list of features for which to add the outliers mean columns.

    Returns:
    pd.DataFrame, pd.DataFrame: The updated train and test datasets.
    """
    for f in features:
        colname = f + '_outliers_mean'
        order_label = train.groupby([f])['outliers'].mean()
        train[colname] = train[f].map(order_label)
        test[colname] = test[f].map(order_label)
    return train, test


def add_derived_features(df):
    """
    Add derived features to a dataframe based on existing features.

    Parameters:
    df (pd.DataFrame): The dataframe to which the derived features will be added.

    Returns:
    pd.DataFrame: The updated dataframe with new derived features.
    """
    for f in ['feature_1', 'feature_2', 'feature_3']:
        df[f'days_{f}'] = df['elapsed_time'] * df[f]
        df[f'days_{f}_ratio'] = df[f] / df['elapsed_time']

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum'] / 3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    df['card_id_total'] = df['hist_card_id_size'] + df['new_card_id_size']
    df['card_id_cnt_total'] = df['hist_card_id_count'] + df['new_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
    df['purchase_amount_total'] = df['hist_cardf_card_id_purchase_amount_sum'] + df[
        'new_cardf_card_id_purchase_amount_sum']
    df['purchase_amount_ratio'] = df['new_cardf_card_id_purchase_amount_sum'] / df[
        'hist_cardf_card_id_purchase_amount_sum']
    df['month_diff_ratio'] = df['new_cardf_month_diff_mean'] / df['hist_cardf_month_diff_mean']
    df['installments_total'] = df['new_cardf_card_id_installments_sum'] + df['auth_cardf_card_id_installments_sum']
    df['installments_ratio'] = df['new_cardf_card_id_installments_sum'] / df['auth_cardf_card_id_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['new_CLV'] = df['new_card_id_count'] * df['new_cardf_card_id_purchase_amount_sum'] / df[
        'new_cardf_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_cardf_card_id_purchase_amount_sum'] / df[
        'hist_cardf_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df

