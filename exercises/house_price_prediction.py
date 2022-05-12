from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna()
    df.drop(df[df['date'] == '0'].index, inplace=True)
    df.drop(df.index[df['price'] <= 0], inplace=True)
    df["year"] = pd.to_datetime(df["date"], format='%Y%m%dT%f').dt.year
    df["years since change"] = df["year"] - df[
        ["yr_built", "yr_renovated"]].max(axis=1)
    dummies = pd.get_dummies(df['year']).add_prefix("year ")
    df = df.join(dummies)
    y_series = df["price"]
    df = df.drop(
        columns=['id', 'lat', 'long', 'date', 'zipcode', 'price', 'year'])
    return df, y_series


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = y.std()
    for feature in X.columns:
        cov_feature_y = X[feature].cov(y)
        std_feature = X[feature].std()
        pearson_corr = cov_feature_y / (std_feature * std_y)
        print(feature, pearson_corr)
        fig = go.Figure().add_trace(
            go.Scatter(x=X[feature], y=y, mode="markers"))
        fig.update_layout(
            title=f'Pearson Correlation between feature\n "{feature}" and response: {round(pearson_corr, 4)}',
            xaxis_title=feature, yaxis_title="prices")
        fig.write_image(output_path + '.' + feature + '.png')

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "corr")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    var_loss = []
    mean = []
    std_loss = []
    lin_reg = LinearRegression()

    for percent in range(10, 101):
        p_loss = []
        for _ in range(10):
            p_x = train_x.sample(frac=percent / 100)
            p_y = train_y.reindex_like((p_x))
            lin_reg.fit(np.asarray(p_x), np.asarray(p_y))
            p_loss += [lin_reg.loss(np.asarray(test_x), np.asarray(test_y))]
        std_loss += [np.std(p_loss)]
        mean += [np.mean(p_loss)]
        var_loss += [np.var(p_loss)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[i / 100 for i in range(10, 101)], y=mean))
    fig.add_trace(
        go.Scatter(x=[i / 100 for i in range(10, 101)],
                   y=np.asarray(mean) + (
                               2 * np.asarray(std_loss)),
                   fill=None, mode="lines", line=dict(color="lightgrey"),
                   showlegend=False))
    fig.add_trace(
        go.Scatter(x=[i / 100 for i in range(10, 101)],
                   y=np.asarray(mean) - (
                               2 * np.asarray(std_loss)),
                   fill='tonexty', mode="lines",
                   line=dict(color="lightgrey"), showlegend=False))
    fig.update_layout(
        title="Mean loss as a function of training by percent",
        xaxis_title="Training percent", yaxis_title="Mean loss")
    fig.show()
