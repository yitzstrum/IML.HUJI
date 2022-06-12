from __future__ import annotations
import numpy as np
import numpy.random
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x_vals = np.random.uniform(-1.2, 2, n_samples)

    y = np.array(
        [(x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2) for x in x_vals])
    y_noise = y + np.random.normal(0, noise, n_samples)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x_vals), pd.Series(y_noise), (2/3))

    x_train_index = np.take(x_vals, train_x.index)
    x_test_index = np.take(x_vals, test_x.index)

    train_x = train_x[0].to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x[0].to_numpy()
    test_y = test_y.to_numpy()

    figure1 = go.Figure(
        [go.Scatter(x=x_vals, y=y, mode="markers", name="Noiseless"),
         go.Scatter(x=x_train_index, y=train_y, mode="markers",
                    name="Train"),
         go.Scatter(x=x_test_index, y=test_y, mode="markers",
                    name="Test")]
        , layout=go.Layout(title=f"Number of samples = {n_samples}, with noise of {noise}",
                           xaxis_title="x", yaxis_title="y"))
    figure1.show()



    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error = []
    validation_error = []

    for k in range(11):
        train_loss, validation_loss = cross_validate(PolynomialFitting(k),
                                                     train_x, train_y,
                                                     mean_square_error)
        train_error.append(train_loss)
        validation_error.append(validation_loss)

    figure2 = go.Figure(
        [go.Scatter(x=list(range(11)), y=train_error, mode="markers+lines",
                    name="Train score"),
         go.Scatter(x=list(range(11)), y=validation_error, mode="markers+lines",
                    name="Validation score")]
        , layout=go.Layout(title=f"Average training and validation error as a function of Polynomial degree. "
                                 f"Number of samples = {n_samples}, with noise of {noise}",
                           xaxis_title="Polynomial degree", yaxis_title="Average error"))
    figure2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_lowest = np.argmin(validation_error)
    poly_k_lowest = PolynomialFitting(k_lowest)
    poly_k_lowest.fit(train_x, train_y)
    test_error = poly_k_lowest.loss(test_x, test_y)
    print(f"The degree of the lowest validation error is {k_lowest} "
          f"and the error is {round(test_error, 2)}")



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples], y[:n_samples]
    test_X, test_y = X[n_samples:], y[n_samples:]


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_val_error, ridge_train_error = [], []
    lasso_val_error, lasso_train_error = [], []
    lambda_vals = np.linspace(0.01, 2, n_samples)

    for lam in lambda_vals:
        ridge_train, ridge_val = cross_validate(RidgeRegression(lam), train_X,
                                                train_y, mean_square_error)
        lasso_train, lasso_val = cross_validate(Lasso(alpha=lam), train_X,
                                                train_y, mean_square_error)
        ridge_train_error.append(ridge_train)
        ridge_val_error.append(ridge_val)
        lasso_val_error.append(lasso_val)
        lasso_train_error.append(lasso_train)

    figure7 = go.Figure(
        [go.Scatter(x=lambda_vals, y=ridge_train_error,
                    mode='lines',
                    marker=dict(color='red'),
                    name="Ridge train error"),
         go.Scatter(x=lambda_vals, y=ridge_val_error,
                    mode='lines',
                    marker=dict(color='black'),
                    name="Ridge validation error"),
         go.Scatter(x=lambda_vals, y=lasso_train_error,
                    mode='lines',
                    marker=dict(color='blue'),
                    name="Lasso train error"),
         go.Scatter(x=lambda_vals, y=lasso_val_error,
                    mode='lines',
                    marker=dict(color='green'),
                    name="Lasso validation error")]
        , layout=go.Layout(title="Average training and validation errors as a function of Lambda", xaxis_title="Lambda",
                           yaxis_title="Average error"))

    figure7.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_arg = lambda_vals[np.argmin(ridge_val_error)]
    best_lasso_arg = lambda_vals[np.argmin(lasso_val_error)]

    ridge_model = RidgeRegression(best_ridge_arg)
    lasso_model = Lasso(alpha=best_lasso_arg)
    linear_model = LinearRegression()

    ridge_model.fit(train_X, train_y)
    lasso_model.fit(train_X, train_y)
    linear_model.fit(train_X, train_y)

    print(f"Ridge test error = {ridge_model.loss(test_X, test_y)} acheived with lambda value of {best_ridge_arg}")
    print(f"Lasso test error = {mean_square_error(test_y, lasso_model.predict(test_X))} acheived with lambda value of {best_lasso_arg}")
    print(f"Linear regression test error = {linear_model.loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()