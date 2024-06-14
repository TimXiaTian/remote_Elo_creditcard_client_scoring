import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, log_loss
import time


def train_model(X, X_test, y, params, folds, model_type='lgb', eval_type='regression'):
    """
    Train and evaluate a model using cross-validation.

    Purpose:
    This function trains LightGBM, XGBoost, or CatBoost models on the given dataset using cross-validation.
    It handles both regression and binary classification tasks.

    Parameters:
    X (np.array): Training data.
    X_test (np.array): Test data.
    y (np.array): Target variable.
    params (dict): Parameters for the model.
    folds (KFold): Cross-validation folds.
    model_type (str): Type of model to train ('lgb', 'xgb', 'cat').
    eval_type (str): Type of evaluation ('regression' or 'binary').

    Returns:
    oof (np.array): Out-of-fold predictions.
    predictions (np.array): Test set predictions.
    scores (list): List of scores for each fold.
    """
    oof = np.zeros(X.shape[0])
    predictions = np.zeros(X_test.shape[0])
    scores = []

    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'lgb':
            trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
            val_data = lgb.Dataset(X[val_idx], y[val_idx])
            clf = lgb.train(params, trn_data, num_boost_round=20000,
                            valid_sets=[trn_data, val_data],
                            verbose_eval=100, early_stopping_rounds=300)
            oof[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        elif model_type == 'xgb':
            trn_data = xgb.DMatrix(X[trn_idx], y[trn_idx])
            val_data = xgb.DMatrix(X[val_idx], y[val_idx])
            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=20000,
                            evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=100, params=params)
            oof[val_idx] = clf.predict(xgb.DMatrix(X[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

        elif model_type == 'cat':
            if eval_type == 'regression':
                clf = CatBoostRegressor(iterations=20000, eval_metric='RMSE', **params)
            elif eval_type == 'binary':
                clf = CatBoostClassifier(iterations=20000, eval_metric='Logloss', **params)

            clf.fit(X[trn_idx], y[trn_idx],
                    eval_set=(X[val_idx], y[val_idx]),
                    cat_features=[], use_best_model=True, verbose=100)

            if eval_type == 'regression':
                oof[val_idx] = clf.predict(X[val_idx])
                predictions += clf.predict(X_test) / folds.n_splits
            elif eval_type == 'binary':
                oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
                predictions += clf.predict_proba(X_test)[:, 1] / folds.n_splits

        print(predictions)

        if eval_type == 'regression':
            scores.append(mean_squared_error(y[val_idx], oof[val_idx]) ** 0.5)
        elif eval_type == 'binary':
            scores.append(log_loss(y[val_idx], oof[val_idx]))

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    return oof, predictions, scores


def train_and_save_model(X_train, X_test, y_train, y_ntrain, y_train_binary, model_params, model_type, eval_type,
                         filename_prefix):
    """
    Train models and save the results.

    Parameters:
    X_train (np.array): Features for the training dataset.
    X_test (np.array): Features for the testing dataset.
    y_train (np.array): Target values for the training dataset.
    y_ntrain (np.array): Target values for the non-outlier training dataset.
    y_train_binary (np.array): Binary target values for the training dataset indicating outliers.
    model_params (dict): Parameters for the model.
    model_type (str): Type of the model ('lgb', 'xgb', or 'cat').
    eval_type (str): Evaluation type ('regression' or 'binary').
    filename_prefix (str): Prefix for the saved files.

    Returns:
    tuple: OOF predictions, test predictions, and scores for the model.
    """
    folds = KFold(n_splits=5, shuffle=True, random_state=4096 if model_type != 'cat' else 18)

    print('=' * 10, 'regression model', '=' * 10)
    oof_model, predictions_model, scores_model = train_model(X_train, X_test, y_train, params=model_params, folds=folds,
                                                             model_type=model_type, eval_type='regression')
    print('=' * 10, 'without outliers regression model', '=' * 10)
    oof_nmodel, predictions_nmodel, scores_nmodel = train_model(X_train, X_test, y_ntrain, params=model_params,
                                                                folds=folds, model_type=model_type,
                                                                eval_type='regression')
    print('=' * 10, 'classification model', '=' * 10)
    model_params[
        'objective'] = 'binary' if model_type == 'lgb' else 'binary:logistic' if model_type == 'xgb' else 'Logloss'
    model_params['metric'] = 'binary_logloss' if model_type in ['lgb', 'xgb'] else 'Logloss'
    oof_bmodel, predictions_bmodel, scores_bmodel = train_model(X_train, X_test, y_train_binary, params=model_params,
                                                                folds=folds, model_type=model_type, eval_type='binary')

    # Save predictions
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df["target"] = predictions_model
    sub_df.to_csv(f'predictions_{filename_prefix}.csv', index=False)

    # Save OOF and predictions
    for name, oof, pred in zip(['oof', 'oof_n', 'oof_b'], [oof_model, oof_nmodel, oof_bmodel],
                               [predictions_model, predictions_nmodel, predictions_bmodel]):
        pd.DataFrame(oof).to_csv(f'./result/{name}_{filename_prefix}.csv', header=None, index=False)
        pd.DataFrame(pred).to_csv(f'./result/predictions_{name}_{filename_prefix}.csv', header=None, index=False)

    return oof_model, predictions_model, scores_model, oof_nmodel, predictions_nmodel, scores_nmodel, oof_bmodel, predictions_bmodel, scores_bmodel
