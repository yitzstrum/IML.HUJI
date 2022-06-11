import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import loss_functions


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    test_error = []
    train_error = []
    for it in range(1, n_learners + 1):
        test_error.append(adaboost.partial_loss(test_X, test_y, it))
        train_error.append(adaboost.partial_loss(train_X, train_y, it))

    go.Figure([
        go.Scatter(x=list(range(1, n_learners)), y=train_error,
                   mode='lines', name='Train Error'),
        go.Scatter(x=list(range(1, n_learners)), y=test_error,
                   mode='lines', name='Test Error')]) \
        .update_layout(
        title="AdaBoost Train and Test error by number of fitted learners",
        xaxis=dict(title="number of fitted learners")).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    figure = make_subplots(2, 2, subplot_titles=[
        f"Number of Iterations = {t}" for t in T])
    for index, t in enumerate(T):
        figure.add_traces([
            decision_surface(lambda v: adaboost.partial_predict(v, t), lims[0],
                             lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       marker=dict(color=test_y),
                       showlegend=False)], rows=int(index / 2) + 1,
            cols=(index % 2) + 1)
    figure.update_layout(
        title="AdaBoost desicion boundary obtained by using the ensemble for the following iterations:")
    figure.show()

    # Question 3: Decision surface of best performing ensemble
    arg_min = np.argmin(np.array(test_error))
    accuracy = loss_functions.accuracy(y_true=test_y,
                                       y_pred=adaboost.partial_predict(test_X,
                                                                       arg_min + 1))
    figure = make_subplots()
    figure.add_traces([
        decision_surface(lambda v: adaboost.partial_predict(v, arg_min + 1),
                         lims[0],
                         lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                   marker=dict(color=test_y),
                   showlegend=False)], rows=1,
        cols=1)
    figure.update_layout(
        title=f"Decision Surface for the Best Performing Ensemble \nFor Adaboost ensemble size of {arg_min + 1} with accuracy of {accuracy}",
        xaxis_title="feature 1", yaxis_title="feature 2")
    figure.show()

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_ / np.max(adaboost.D_) * 5
    figure4 = go.Figure()
    figure4.add_traces([
        decision_surface(adaboost._predict,
                         lims[0],
                         lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                   showlegend=False,
                   marker=dict(size=D, color=train_y,
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))])
    figure4.update_layout(
        title=f"Training set with points proportional to weight value",
        xaxis_title="feature 1", yaxis_title="feature 2")
    figure4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
