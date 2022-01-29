# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from correlation_matrix import plot_correlation_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

# %%
EXPERIMENT = 1
EXPERIMENT_PATH = 'exp_1_effective_porosity' if EXPERIMENT == 1 else 'exp_2_total_porosity'
DATA_FILE = f'../results/{EXPERIMENT_PATH}/feature_selection/feature_selected_data.csv'
PATH_SAVE_FILES = f'../results/{EXPERIMENT_PATH}/feature_selection/correlation/'
Path(PATH_SAVE_FILES).mkdir(parents=True, exist_ok=True)
# %%
# Import the data
dataset = pd.read_csv(DATA_FILE, index_col=0)
dataset_mean = dataset.groupby(by='sample_name').mean()
dataset.describe()
# %%
# Plots the correlation matrix as a heatmap of all the features
POROSITY_COLUMN_NAME = dataset.columns[-1]
DATASET_COLUMNS = list(dataset.columns)


# %%

def create_graphs(reflectance_value,
                  porosity_value,
                  wavelength_name,
                  save_evaluation=False):
    '''
    Creates scatter and residual plot of predictions passed in the
    first two parameters.

            Parameters:
                    y_array (NumpyArray): Ground Truth value
                    y_hat_array (NumpyArray): Estimated Values
                    wavelength_name (String): Name of the wavelength_name
                    path_save_evaluation (String): Path to save graphs and
                    metrics

            Returns:
                    None
    '''
    plt.clf()
    plt.style.use(['seaborn-ticks'])
    plt.figure(figsize=(6.5, 4.1))  # 4.75))

    # Plots the estimatives
    plt.plot(reflectance_value, porosity_value, "o")

    # Plots a linear fit between prediction and actual value
    linear = LinearRegression()
    linear.fit(reflectance_value, porosity_value)
    plt.plot(reflectance_value, linear.predict(
        reflectance_value), '-', color='red')

    r, _ = pearsonr(reflectance_value.flatten(), porosity_value.flatten())
    plt.grid(True)
    plt.ylabel('Laboratory Determined Porosity [%]')
    plt.xlabel(f'Normalized Reflectance {wavelength_name}nm - r:  {r:7.4f}')
    if save_evaluation:

        # Save Graph
        name_in_graph = wavelength_name.replace('.', '_')
        plt.savefig(f'{PATH_SAVE_FILES}{name_in_graph}.png',
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()


# %%
for wavelength in DATASET_COLUMNS[1:-1]:
    print(wavelength)
    create_graphs(dataset_mean[wavelength].values.reshape(-1, 1),
                  dataset_mean[POROSITY_COLUMN_NAME].values.reshape(-1, 1),
                  wavelength,
                  save_evaluation=True)
