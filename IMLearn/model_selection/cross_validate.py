from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = 0, 0
    split_x = np.array_split(X, cv)
    split_y = np.array_split(y, cv)
    for k in range(cv):
        train_x = np.concatenate((split_x[:k] + split_x[k+1:]), axis=0)
        train_y = np.concatenate((split_y[:k] + split_y[k+1:]), axis=0)
        estimator.fit(train_x, train_y)
        validate_x = split_x[k]
        validate_y = split_y[k]
        validation_score += scoring(validate_y, estimator.predict(validate_x))
        train_score += scoring(train_y, estimator.predict(train_x))

    return train_score / cv, validation_score / cv


