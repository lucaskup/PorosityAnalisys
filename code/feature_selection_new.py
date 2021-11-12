# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from correlation_matrix import plot_correlation_matrix


# %%
dataset = pd.read_csv(f'../data/more_data.csv',
                      sep=';',
                      decimal='.')


# %%
dataset = dataset.drop(['seq', 'place', 'file_name'], axis=1)
visual_selection = ['422.4', '486.3', '670', '970.3', '1005.4', '1412.8',
                    '1461.8', '1900.9', '1932.2', '2151.3', '2222.1', '2324.3']
# %%
POROSITY_COLUMN_NAME = dataset.columns[-1]
DATASET_COLUMNS = list(dataset.columns[1:])
DATASET_COLUMNS = DATASET_COLUMNS[50:-51] + [POROSITY_COLUMN_NAME]
DIVISIONS_IN_SPECTRA = 20
WAVELENGTHS_PER_DIVISION = len(DATASET_COLUMNS) // DIVISIONS_IN_SPECTRA
for i in range(DIVISIONS_IN_SPECTRA):
    lower_index_cut = i * WAVELENGTHS_PER_DIVISION
    lower_column_name = DATASET_COLUMNS[lower_index_cut]
    upper_index_cut = lower_index_cut + WAVELENGTHS_PER_DIVISION - 1
    upper_column_name = DATASET_COLUMNS[upper_index_cut]
    partial_dataset = dataset[DATASET_COLUMNS[lower_index_cut:upper_index_cut] +
                              [POROSITY_COLUMN_NAME]]

    filePathAndName = f'../results/featureSelection/CorrMatrix_Wav_{lower_column_name}_{upper_column_name}'
    partial_correlation_matrix = plot_correlation_matrix(partial_dataset,
                                                         annotate_cells=False,
                                                         file_name=f'{filePathAndName}.png')

    corr_above_07 = partial_correlation_matrix[partial_correlation_matrix[POROSITY_COLUMN_NAME
                                                                          ].abs() > 0.45][POROSITY_COLUMN_NAME]
    print('Best Ones:\n', corr_above_07)
    corr_above_07.to_csv(f'{filePathAndName}.csv')
# %%
correlation_matrix_selection = ['818.1', '940.5', '996.2',
                                '1932.2', '2057.2', '2156.5', '2269', '2329.1']
# %%
selected_features_all_methods = list(
    dict.fromkeys(visual_selection + correlation_matrix_selection))
selected_features_all_methods = list(
    sorted(selected_features_all_methods, key=lambda x: float(x)))
# %%
feature_selected_data = dataset[['sample_name'] + selected_features_all_methods +
                                [dataset.columns[-1]]]
plot_correlation_matrix(feature_selected_data.drop(columns='sample_name'),
                        file_name='../results/featureSelection/CorrMatrix_Selected_Wav_png')
print(feature_selected_data.columns)
# %%
feature_selected_data.to_csv(
    '../results/featureSelection/featureSelectedData.csv')