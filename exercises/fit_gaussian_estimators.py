from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, size=1000)
    uni = UnivariateGaussian().fit(samples)
    print(uni.mu_)
    print(uni.var_)

    # Question 2 - Empirically showing sample mean is consistent
    abs_dist = []
    for size in range(10, 1001, 10):
        abs_dist.append(abs(10 - np.mean(samples[:size])))
    sample_sizes = [i for i in range(10, 1001, 10)]
    go.Figure([go.Scatter(x=sample_sizes, y=abs_dist, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\mu-\hat\mu|$",
                  height=800)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    samples_pdf = uni.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=samples_pdf, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - sample values}$",
                  yaxis_title="r$PDFs$",
                  height=800)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, size=1000)
    multi = MultivariateGaussian().fit(samples)
    print(multi.mu_)
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    f1_f3 = np.linspace(-10, 10, 200)
    all_options = np.transpose(np.array(np.meshgrid(f1_f3, 0, f1_f3, 0))).reshape(-1, 4)
    log_likelihood_func = lambda mu_: MultivariateGaussian.log_likelihood(mu_, cov, samples)
    likelihood = np.transpose(np.apply_along_axis(log_likelihood_func, 1, all_options).reshape(200, 200))
    go.Figure(data=[go.Heatmap(x=f1_f3, y=f1_f3, z=likelihood, type='heatmap')],
              layout=go.Layout(
                  title="Heatmap Of Log-Likelihood Models according to f1 and f3 Values",
                  xaxis_title="f3",
                  yaxis_title="f1",
                  )).show()

    # Question 6 - Maximum likelihood
    location = np.where(likelihood == np.amax(likelihood))
    row_max = location[0][0]
    col_max = location[1][0]
    print('(' + str(f1_f3[row_max]) + ', ' + str(f1_f3[col_max]) + ')')
    print(np.amax(likelihood))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
