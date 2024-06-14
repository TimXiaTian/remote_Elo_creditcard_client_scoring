import pandas as pd
import numpy as np
import datetime

from pandas import DataFrame
from sklearn.model_selection import KFold
from data_processing import (reduce_mem_usage, change_object_cols, merge_first_act_month, gen_date_features,
                             month_trans, week_trans, get_expand_common)
from feature_optimization import (aggregate_transactions, get_quantile, get_cardf_tran, get_cardf_tran_last2,
                                  successive_aggregates, merge_datasets, create_outliers_feature, add_outliers_mean,
                                  add_derived_features)
from feature_filting import (get_columns_to_delete, load_and_process_data)
from model_training import train_model, train_and_save_model
from model_ensemble import stack_model


def main():
    # load datasets
    print('------')
    print('Loading data...')
    new_transactions = pd.read_csv('data/new_merchant_transactions.csv', parse_dates=['purchase_date'])
    historical_transactions = pd.read_csv('data/historical_transactions.csv', parse_dates=['purchase_date'])
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Step1: data_processing
    print('processing training data and testing data...')
    for df_name in ['train', 'test']:
        df = globals()[df_name]
        df = gen_date_features(df)
        globals()[df_name] = df

    print('processing historical_transactions and new_transactions data...')
    for df_name in ['historical_transactions', 'new_transactions']:
        df = globals()[df_name]

        for col in ['authorized_flag', 'category_1']:
            df[col] = change_object_cols(df[col])
        df = merge_first_act_month(df, train, test)
        df = get_expand_common(df)

        globals()[df_name] = df

    # Step2: feature_optimization
    print('\n ------')
    print('optimizing features for historical_transactions and new_transactions data..')

    print(f'\t generate statistics features...')
    for df_name in ['historical_transactions', 'new_transactions']:
        df = globals()[df_name]
        if df_name == 'historical_transactions':
            print('\t generate statistics features for historical transactions...')

            auth_base_stat = aggregate_transactions(df[df['authorized_flag'] == 1],
                                                    prefix='auth_')
            hist_base_stat = aggregate_transactions(df[df['authorized_flag'] == 0],
                                                    prefix='hist_')

            globals()['auth_base_stat'] = auth_base_stat
            globals()['hist_base_stat'] = hist_base_stat

        elif df_name == 'new_transactions':
            print('\t generate statistics features for new transactions...')

            new_base_stat = aggregate_transactions(df, prefix='new_')

            globals()['new_base_stat'] = new_base_stat

    print('')
    print('generating card features...')

    print('\t auth...')
    authorized_transactions = historical_transactions.loc[historical_transactions['authorized_flag'] == 1]
    auth_cardf_tran = get_cardf_tran(authorized_transactions, 3, prefix='auth_cardf_')
    print('\t hist...')
    hist_cardf_tran = get_cardf_tran(historical_transactions, 3, prefix='hist_cardf_')
    print('\t new...')
    reference_days = historical_transactions.groupby('card_id')['purchase_date'].last().to_frame('reference_day')
    reference_days.reset_index(inplace=True)
    new_transactions = new_transactions.merge(reference_days, on='card_id', how='left')
    new_cardf_tran = get_cardf_tran(new_transactions, 5, prefix='new_cardf_')

    print('')
    print('generating card features for last 2 months...')
    print('\t hist_last2...')
    hist_cardf_tran_last2 = get_cardf_tran_last2(historical_transactions, month=3, prefix='hist_last2_')

    print('')
    print('generate second-order cross-features...')
    print('\t hist_secondorder2...')
    hist_levelAB = successive_aggregates(historical_transactions, prefix='hist_levelAB_')

    print('')
    print('merging all of our derivative features into our training set and testing set...')
    datasets_base_stat = [auth_base_stat, hist_base_stat, new_base_stat]
    datasets_cardf_tran = [auth_cardf_tran, hist_cardf_tran, new_cardf_tran, hist_cardf_tran_last2, hist_levelAB]
    key = 'card_id'

    print('#_____statistic features')
    train, test = merge_datasets(train, test, datasets_base_stat, [key] * 3)

    print('#_____global card id features')
    train, test = merge_datasets(train, test, datasets_cardf_tran[:-1], [key] * 3)

    print('#_____last 2 months card id features')
    train, test = merge_datasets(train, test, [datasets_cardf_tran[-2]], [key])

    print('#_____second-order cross features')
    train, test = merge_datasets(train, test, [datasets_cardf_tran[-1]], [key])

    print('')
    print('deriving some simple operation features ')
    train = create_outliers_feature(train)
    train, test = add_outliers_mean(train, test, ['feature_1', 'feature_2', 'feature_3'])
    for df_name in ['train', 'test']:
        df = globals()[df_name]
        df = add_derived_features(df)
        globals()[df_name] = df

    # Step3: feature_filting
    print('\n ------')
    print('filting our features...')

    del_cols = []
    for col in train.columns:
        if 'subsector_id_cnt_' in col and 'new_cardf':
            del_cols.append(col)
    del_cols1 = []
    for col in train.columns:
        if 'subsector_id_cnt_' in col and 'hist_last2_' in col:
            del_cols1.append(col)
    del_cols2 = []
    for col in train.columns:
        if 'subsector_id_cnt_' in col and 'auth_cardf' in col:
            del_cols2.append(col)
    del_cols3 = []
    for col in train.columns:
        if 'merchant_category_id_month_lag_nunique_' in col and '_pivot_supp' in col:
            del_cols3.append(col)
        if 'city_id' in col and '_pivot_supp' in col:
            del_cols3.append(col)
        if 'month_diff' in col and 'hist_last2_' in col:
            del_cols3.append(col)
        if 'month_diff_std' in col or 'month_diff_gap' in col:
            del_cols3.append(col)
    fea_cols = [col for col in train.columns if train[col].dtypes != 'object' and train[
        col].dtypes != '<M8[ns]' and col != 'target' not in col and col != 'min_num' \
                and col not in del_cols and col not in del_cols1 and col not in del_cols2 and col != 'target1' \
                and col != 'card_id_cnt_ht_pivot_supp' and col not in del_cols3]

    print('\t before deleting:', train.shape[1])
    print('\t after deleting:', len(fea_cols))

    train = train[fea_cols + ['target']]
    fea_cols.remove('outliers')
    test = test[fea_cols]

    train.to_csv('./data/all_train_features.csv', index=False)
    test.to_csv('./data/all_test_features.csv', index=False)

    print('')
    print('Preparing target variables for modeling...')

    train_path = './data/all_train_features.csv'
    test_path = './data/all_test_features.csv'
    inf_cols = ['new_cardf_card_id_cnt_divide_installments_nunique',
                'hist_last2_card_id_cnt_divide_installments_nunique']

    train, test, ntrain, target, ntarget, target_binary = load_and_process_data(train_path, test_path, inf_cols)

    y_train = target
    y_ntrain = ntarget
    y_train_binary = target_binary

    # Step4: modeling
    print('------')
    print('Training our models...')

    lgb_params = {'num_leaves': 63,
                  'min_data_in_leaf': 32,
                  'objective': 'regression',
                  'max_depth': -1,
                  'learning_rate': 0.01,
                  "min_child_samples": 20,
                  "boosting": "gbdt",
                  "feature_fraction": 0.9,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.9,
                  "bagging_seed": 11,
                  "metric": 'rmse',
                  "lambda_l1": 0.1,
                  "verbosity": -1}

    xgb_params = {'eta': 0.05,
                  'max_leaves': 47,
                  'max_depth': 10,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'min_child_weight': 40,
                  'max_bin': 128,
                  'reg_alpha': 2.0,
                  'reg_lambda': 2.0,
                  'objective': 'reg:linear',
                  'eval_metric': 'rmse',
                  'silent': True,
                  'nthread': 4}

    cat_params = {'learning_rate': 0.05,
                  'depth': 9,
                  'l2_leaf_reg': 10,
                  'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter',
                  'od_wait': 50,
                  'random_seed': 11,
                  'allow_writing_files': False}

    X_ntrain = ntrain[fea_cols].values
    X_train = train[fea_cols].values
    X_test = test[fea_cols].values

    print('LightGBM modeling...')
    oof_lgb, predictions_lgb, scores_lgb, oof_nlgb, predictions_nlgb, scores_nlgb, oof_blgb, predictions_blgb, scores_blgb = train_and_save_model(
        X_train, X_test, y_train, y_ntrain, y_train_binary, lgb_params, 'lgb', 'regression', 'lgb')

    print('XGBoost modeling...')
    oof_xgb, predictions_xgb, scores_xgb, oof_nxgb, predictions_nxgb, scores_nxgb, oof_bxgb, predictions_bxgb, scores_bxgb = train_and_save_model(
        X_train, X_test, y_train, y_ntrain, y_train_binary, xgb_params, 'xgb', 'regression', 'xgb')

    print('CatBoost modeling...')
    oof_cat, predictions_cat, scores_cat, oof_ncat, predictions_ncat, scores_ncat, oof_bcat, predictions_bcat, scores_bcat = train_and_save_model(
        X_train, X_test, y_train, y_ntrain, y_train_binary, cat_params, 'cat', 'regression', 'cat')

    # Combine predictions
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df["target"] = (predictions_lgb + predictions_xgb.flatten() + predictions_cat.flatten()) / 3
    sub_df.to_csv('combined_predictions.csv', index=False)

    # Step5: model ensemble
    print('------')
    print('ensemble our models...')

    print('\t weight ensemble...')
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df["target"] = (predictions_lgb + predictions_xgb.values.flatten() + predictions_cat.values.flatten()) / 3
    sub_df.to_csv('predictions_wei_average.csv', index=False)

    print('\t stacking...')
    print('=' * 30)
    oof_stack, predictions_stack = stack_model(oof_lgb, oof_xgb, oof_cat, predictions_lgb, predictions_xgb,
                                               predictions_cat, target)
    print('=' * 30)
    oof_nstack, predictions_nstack = stack_model(oof_nlgb, oof_nxgb, oof_ncat, predictions_nlgb, predictions_nxgb,
                                                 predictions_ncat, ntarget)
    print('=' * 30)
    oof_bstack, predictions_bstack = stack_model(oof_blgb, oof_bxgb, oof_bcat, predictions_blgb, predictions_bxgb,
                                                 predictions_bcat, target_binary, eval_type='binary')

    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df["target"] = predictions_stack
    sub_df.to_csv('predictions_stack.csv', index=False)

    print('\t Trick ensemble...')
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df["target"] = predictions_bstack * -33.219281 + (1 - predictions_bstack) * predictions_nstack
    sub_df.to_csv('predictions_trick.csv', index=False)

    print('\t Trick stacking...')
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df["target"] = ((predictions_bstack * -33.219281 + (1 - predictions_bstack) * predictions_nstack)
                        * 0.5 + predictions_stack * 0.5)
    sub_df.to_csv('predictions_trick&stacking.csv', index=False)


if __name__ == "__main__":
    main()
