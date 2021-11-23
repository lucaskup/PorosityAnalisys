# %%
from scipy.special import comb
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
mpl.rcParams['figure.dpi'] = 400

# %%
EXPERIMENT = 1
EXPERIMENT_PATH = 'exp_1_effective_porosity' if EXPERIMENT == 1 else 'exp_2_total_porosity'
DATA_FILE = f'../results/{EXPERIMENT_PATH}/feature_selection/feature_selected_data.csv'
PATH_SAVE_FILES = f'../results/{EXPERIMENT_PATH}/model_trained/'
# %%
# Import the data
dataset = pd.read_csv(DATA_FILE, index_col=0)
dataset.describe()
# %%
dataset = dataset.groupby(by='sample_name').mean()

# %%
X = dataset.values[:, :-1].astype(np.float64)
#groups = dataset.values[:, 0].astype(np.str)
Y = dataset['Porosity (%)'].values.astype(np.float64)
# %%
scX = StandardScaler()
X = scX.fit_transform(X)
mmY = MinMaxScaler()
Y = mmY.fit_transform(Y.reshape(-1, 1)).ravel()

# %%
n_samples_data = len(X)
indices = np.arange(start=0, stop=n_samples_data).reshape(-1, 1)
X_ = np.concatenate([indices, X], axis=1)
results = []
for i in range(2, n_samples_data-1):
    bootstrap_samples = min(1000, int(comb(n_samples_data, i)))
    print(
        f'Experiment:\n Train Size: {i}\n Bootstrap Samples: {bootstrap_samples}')
    mse_train = 0
    mse_test = 0
    for j in range(bootstrap_samples):
        X_bootstrap = resample(X_, replace=False, n_samples=i)
        index_train_bootstrap = X_bootstrap[:, 0].astype(np.int)
        X_bootstrap_train = X_bootstrap[:, 1:]
        Y_bootstrap_train = Y[index_train_bootstrap]

        X_bootstrap_test = np.delete(X, index_train_bootstrap, 0)
        Y_bootstrap_test = np.delete(Y, index_train_bootstrap, 0)

        elasticNet = ElasticNet(alpha=0.00025, l1_ratio=1, max_iter=100000)
        # elasticNet = SVR(gamma=0.0011,
        #                 C=50,
        #                 epsilon=0.05,
        #                 kernel='rbf')
        elasticNet = elasticNet.fit(X_bootstrap_train, Y_bootstrap_train)

        train_predictions = mmY.inverse_transform(
            elasticNet.predict(X_bootstrap_train).reshape(-1, 1))
        train_ground_truth = mmY.inverse_transform(
            Y_bootstrap_train.reshape(-1, 1))
        mse_train += mean_squared_error(train_predictions,
                                        train_ground_truth)

        test_predictions = mmY.inverse_transform(
            elasticNet.predict(X_bootstrap_test).reshape(-1, 1))
        test_ground_truth = mmY.inverse_transform(
            Y_bootstrap_test.reshape(-1, 1))
        mse_test += mean_squared_error(test_predictions,
                                       test_ground_truth)

    print(mse_train/bootstrap_samples, mse_test/bootstrap_samples)
    results.append((i, mse_train/bootstrap_samples,
                   mse_test/bootstrap_samples))
# %%
results = np.asfarray(results)
plt.figure(figsize=(6, 4))
plt.plot(results[:, 0], results[:, 1], label='train')
plt.plot(results[:, 0], results[:, 2], label='validation')

plt.style.use(['seaborn-ticks'])
plt.grid(True)
plt.xlabel('Rock Samples in Training Set', fontsize=11)
plt.ylabel('MSE [Porosity²]', fontsize=11)
plt.legend()
plt.title('Learning Curve Elastic Net')

plt.savefig(f'{PATH_SAVE_FILES}LearningCurve.png',
            dpi=450)
