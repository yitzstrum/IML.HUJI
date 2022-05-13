from __future__ import annotations
from typing import Tuple, NoReturn
import IMLearn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_err = 1
        for feature in range(X.shape[1]):
            for sign in {-1, 1}:
                threshold, curr_err = self._find_threshold(X[:, feature], y, sign)
                if curr_err < min_err:
                    min_err = curr_err
                    self.threshold_ = threshold
                    self.j_ = feature
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        test_feature_array = X[:, self.j_]
        return np.array([-self.sign_ if sample < self.threshold_ else self.sign_ for sample in test_feature_array])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # todo: fix function!!
        sorted_values = np.sort(values)
        indices = np.argsort(values)
        sorted_labels = np.take(labels, indices)
        loss = []
        res = np.ones(len(labels)) * sign
        loss.append(np.sum(np.where((res * -sign) != np.sign(sorted_labels),
                                    np.abs(sorted_labels), 0)))
        loss.append(np.sum(
            np.where(res != np.sign(sorted_labels), np.abs(sorted_labels), 0)))
        i = 0
        un = np.unique(sorted_values)
        for v in un:
            while i < len(sorted_values) and sorted_values[i] == v:
                res[i] = -sign
                i += 1
            loss.append(np.sum(
                np.where(res != np.sign(sorted_labels), np.abs(sorted_labels),
                         0)))
        idx = np.argmin(np.array(loss))
        if idx > 1:
            return un[idx - 2], loss[idx] / len(sorted_labels)
        if idx == 1:
            return un[-1] + 1, loss[idx] / len(sorted_labels)
        if idx == 0:
            return un[0] - 1, loss[idx] / len(sorted_labels)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return IMLearn.metrics.misclassification_error(y, self._predict(X))
