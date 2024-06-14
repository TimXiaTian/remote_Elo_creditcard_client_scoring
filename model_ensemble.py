import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import RepeatedKFold


def stack_model(oof_1, oof_2, oof_3, predictions_1, predictions_2, predictions_3, y, eval_type='regression'):
    """
    Perform stacking ensemble using Bayesian Ridge regression.

    Purpose:
    This function combines out-of-fold predictions and test set predictions from multiple models
    using Bayesian Ridge regression to produce final predictions.

    Parameters:
    oof_1 (np.array): Out-of-fold predictions from the first model.
    oof_2 (np.array): Out-of-fold predictions from the second model.
    oof_3 (np.array): Out-of-fold predictions from the third model.
    predictions_1 (np.array): Test set predictions from the first model.
    predictions_2 (np.array): Test set predictions from the second model.
    predictions_3 (np.array): Test set predictions from the third model.
    y (np.array): True target values.
    eval_type (str): Type of evaluation, either 'regression' or 'binary'.

    Returns:
    oof (np.array): Out-of-fold predictions from the stacking model.
    predictions (np.array): Final test set predictions from the stacking model.
    """
    # Part 1: Data preparation
    # Concatenate out-of-fold predictions and test predictions
    train_stack = np.hstack([oof_1, oof_2, oof_3])
    test_stack = np.hstack([predictions_1, predictions_2, predictions_3])

    oof = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    # Part 2: Multiple rounds of cross validation
    folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2020)

    # Cross-validation loop
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, y)):
        print("fold n°{}".format(fold_ + 1))

        trn_data, trn_y = train_stack[trn_idx], y[trn_idx]
        val_data, val_y = train_stack[val_idx], y[val_idx]

        print("-" * 10 + "Stacking " + str(fold_ + 1) + "-" * 10)

        # Use Bayesian Ridge regression as the final model
        clf = BayesianRidge()
        clf.fit(trn_data, trn_y)

        # Predict validation data and update oof predictions
        oof[val_idx] = clf.predict(val_data)

        # Predict test data and average predictions across folds
        predictions += clf.predict(test_stack) / (5 * 2)

    # Evaluate the stacking model
    if eval_type == 'regression':
        print('Mean RMSE: ', np.sqrt(mean_squared_error(y, oof)))
    elif eval_type == 'binary':
        print('Mean Log Loss: ', log_loss(y, oof))

    return oof, predictions