import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df['Temp'] > -40]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_X = X[X['Country'] == 'Israel']
    israel_X['Year'] = israel_X['Year'].astype(str)
    fig = px.scatter(israel_X, x=israel_X['DayOfYear'], y=israel_X['Temp'],
                     color="Year")
    fig.update_layout(
        title='Temperature according to Day of year',
        xaxis_title="Day of year", yaxis_title="Temp")
    fig.show()

    monthly_std = israel_X.groupby('Month')['Temp'].std().rename(
        "Standard deviation")
    fig = px.bar(monthly_std, title="Standard deviation of each month")
    fig.show()

    # Question 3 - Exploring differences between countries
    group = X.groupby(['Country', 'Month'])['Temp'].agg(
        ['mean', 'std']).reset_index()
    px.line(group, x='Month', y='mean', line_group='Country', color='Country',
            error_y='std').show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(israel_X['DayOfYear'], israel_X['Temp'])
    loss = []
    for k in range(1, 11):
        poly_regressor = PolynomialFitting(k)
        poly_regressor.fit(np.asarray(train_x), np.asarray(train_y))
        loss.append(round(poly_regressor.loss(np.asarray(test_x), np.asarray(test_y)), 2))
        print("Test Error recorded for K = ", k, ": ", loss[k-1])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[i for i in range(1, 11)], y=loss))
    fig.update_layout(title='Test Error according to K', xaxis_title='K Degree',
                      yaxis_title='Test Error')
    fig.show()


    # Question 5 - Evaluating fitted model on different countries
    israel_model_5 = PolynomialFitting(5)
    israel_model_5.fit(israel_X['DayOfYear'], israel_X['Temp'])
    countries = ["Jordan", "South Africa", "The Netherlands"]
    loss_data = [israel_model_5.loss(X[X.Country == country]['DayOfYear'], X[X.Country == country]['Temp']) for country in countries]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=countries, y=loss_data))
    fig.update_layout(title="Test Error of model fitted for Israel apply to other counties according to country",
                      xaxis_title='Country',
                      yaxis_title="Test Error")
    fig.show()