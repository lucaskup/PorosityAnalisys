from sklearn.model_selection import cross_validate

import pandas as pd
import numpy as np

from scipy.stats import t
from sklearn.model_selection import RepeatedKFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

RESIDUALS_FILE_NAME = '../results/modelComparison/residualsUnder10Rep10foldCV.csv'


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : int
        Variance-corrected standard deviation of the set of differences.
    """
    n = n_train + n_test
    corrected_var = (
        np.var(differences, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def execute_experiment():
    linear = LinearRegression()
    ridge = Ridge(alpha=0.1, max_iter=100)
    lasso = Lasso(alpha=0.00025, max_iter=1000)
    elasticNet = ElasticNet(alpha=0.00025, l1_ratio=1, max_iter=1000)
    knn = KNeighborsRegressor(n_neighbors=2, metric='minkowski')
    svr = SVR(gamma=5, C=10, epsilon=0.01, kernel='rbf')
    forest = RandomForestRegressor(n_estimators=500, criterion='mae')
    mlp = MLPRegressor(max_iter=3000, hidden_layer_sizes=(20, 15, 15, 10),
                       activation='relu', alpha=0.001,
                       learning_rate='adaptive', learning_rate_init=0.001,
                       batch_size=3, solver='adam')

    results = list(map(lambda model: cross_validate(model,
                                                    X,
                                                    y=Y,
                                                    cv=cv,
                                                    scoring='neg_mean_squared_error'),
                       [linear, ridge, lasso, elasticNet, knn, forest, svr, mlp]))
    boxplot_data = list(map(lambda x: x['test_score'], results))

    plt.boxplot(boxplot_data)
    plt.show()

    linear_results, ridge_results, lasso_results, \
        elasticnet_results, knn_results, forest_results, \
        svr_results, mlp_results = boxplot_data

    residuals_df = pd.DataFrame({'Linear': linear_results,
                                'Lasso': lasso_results,
                                 'Ridge': ridge_results,
                                 'ElasticNet': elasticnet_results,
                                 'KNN': knn_results,
                                 'RF': forest_results,
                                 'SVR': svr_results,
                                 'MLP': mlp_results})

    residuals_df.to_csv(RESIDUALS_FILE_NAME,
                        sep=';', decimal='.')


# Import the data
cv = RepeatedKFold(
    n_splits=10, n_repeats=10, random_state=0
)

dataset = pd.read_csv('../results/featureSelection/featureSelectedData.csv',
                      index_col=0)

X = dataset.values[:, :-1].astype(np.float64)
Y = dataset['Porosity (%)'].values.astype(np.float64)

mmY = MinMaxScaler()
Y = mmY.fit_transform(Y.reshape(-1, 1)).ravel()

# execute_experiment()

residual = pd.read_csv(
    RESIDUALS_FILE_NAME, sep=';', decimal='.')
model_1_scores = residual['SVR'].values  # scores of the best model
# scores of the second-best model
model_2_scores = residual['ElasticNet'].values

differences = model_1_scores - model_2_scores

n = 10  # number of test sets
df = n - 1
n_train = len(list(cv.split(X, Y))[0][0])
n_test = len(list(cv.split(X, Y))[0][1])

t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
print(f"Corrected t-value: {t_stat:.3f}\n"
      f"Corrected p-value: {p_val:.3f}")

t_stat_uncorrected = (
    np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
)
p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)

print(f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
      f"Uncorrected p-value: {p_val_uncorrected:.3f}")
