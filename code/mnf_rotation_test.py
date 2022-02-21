# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pysptools.noise import MNF, Whiten

from spectral import calc_stats, noise_from_diffs, mnf

from correlation_matrix import plot_correlation_matrix

# %%
EXPERIMENT = 1
EXPERIMENT_PATH = 'exp_1_effective_porosity' if EXPERIMENT == 1 else 'exp_2_total_porosity'
DATA_FILE = 'exp_1_effective_porosity.csv' if EXPERIMENT == 1 else 'exp_2_total_porosity_reflec.csv'
DATA_FILE = 'data.csv' if EXPERIMENT == 1 else 'exp_2_total_porosity_reflec.csv'
PATH_SAVE_FILES = f'../results/{EXPERIMENT_PATH}/feature_selection/'

# %%
dataset = pd.read_csv(f'../data/exp_1_effective_porosity_reflectance.csv',
                      sep=';',
                      decimal='.')

# %%
hsi_cube = dataset.values
hsi_cube = hsi_cube[:, 4:-1]
hsi_cube = hsi_cube.astype(np.float32)
# %%
hsi_cube = hsi_cube.reshape(235, 1018)
hsi_cube_to_process = np.ones((235, 235, 1018))
for i in range(235):
    hsi_cube_to_process[i, :, :] = hsi_cube
#hsi_cube_1 = hsi_cube[0,:,:,:]
# %%
proc = MNF()
mnf_processed = proc.apply(hsi_cube_to_process)
# %%
mnf_spectra = mnf_processed[0, :, :]
np.savetxt('teste0402.csv', mnf_spectra, delimiter=';', fmt='%10.5f')

# %%
inverse_transformed_spectra = proc.inverse_transform(mnf_processed)
# %%
wt = Whiten()
wt.apply(hsi_cube_to_process)
# %%
dataset = dataset.drop(['seq', 'place'], axis=1)
visual_selection = ['422.4', '486.3', '670', '970.3', '1005.4', '1412.8',
                    '1461.8', '1900.9', '1932.2', '2151.3', '2222.1', '2324.3']
# %%
POROSITY_COLUMN_NAME = dataset.columns[-1]
DATASET_COLUMNS = list(dataset.columns[1:])
DATASET_COLUMNS = DATASET_COLUMNS[50:-51] + [POROSITY_COLUMN_NAME]
DIVISIONS_IN_SPECTRA = 20
WAVELENGTHS_PER_DIVISION = len(DATASET_COLUMNS) // DIVISIONS_IN_SPECTRA
# %%
for i in range(DIVISIONS_IN_SPECTRA):
    lower_index_cut = i * WAVELENGTHS_PER_DIVISION
    lower_column_name = DATASET_COLUMNS[lower_index_cut]
    upper_index_cut = lower_index_cut + WAVELENGTHS_PER_DIVISION - 1
    upper_column_name = DATASET_COLUMNS[upper_index_cut]
    partial_dataset = dataset[DATASET_COLUMNS[lower_index_cut:upper_index_cut] +
                              [POROSITY_COLUMN_NAME]].astype(float)

    filePathAndName = f'{PATH_SAVE_FILES}CorrMatrix_Wav_{lower_column_name}_{upper_column_name}'
    partial_correlation_matrix = plot_correlation_matrix(partial_dataset,
                                                         annotate_cells=False,
                                                         file_name=f'{filePathAndName}.png')

    # print(partial_correlation_matrix)
    corr_above_07 = partial_correlation_matrix[partial_correlation_matrix[POROSITY_COLUMN_NAME
                                                                          ].abs() > 0.45][POROSITY_COLUMN_NAME]
    print('Best Ones:\n', corr_above_07)
    corr_above_07.to_csv(f'{filePathAndName}.csv')
# %%
if EXPERIMENT == 1:
    correlation_matrix_selection = ['853.1', '940.5', '996.2',
                                    '1842.3', '1932.2', '2057.2', '2166.7', '2352.6']
else:
    correlation_matrix_selection = ['855.5', '940.5', '996.2', '1209.1', '1401.4', '1887.2',
                                    '1932.2', '2019.8', '2197.1', '2222.1', '2234.6']
    correlation_matrix_selection = ['843.6', '940.5', '996.8', '1439.2', '1890.6',
                                    '1926.6', '2019.8', '2109.9', '2222.1', '2281.1']
    #correlation_matrix_selection = DATASET_COLUMNS[:-1]

# %%
selected_features_all_methods = list(
    dict.fromkeys(visual_selection + correlation_matrix_selection))
selected_features_all_methods = list(
    sorted(selected_features_all_methods, key=lambda x: float(x)))
# %%
feature_selected_data = dataset[['sample_name'] + selected_features_all_methods +
                                [dataset.columns[-1]]]
# feature_selected_data = feature_selected_data.groupby(
#    by='sample_name').mean().round(4)
# %%
plot_correlation_matrix(feature_selected_data,
                        file_name=f'{PATH_SAVE_FILES}CorrMatrix_Selected_Wav_png')
print(feature_selected_data.columns)
# %%
feature_selected_data.to_csv(
    f'{PATH_SAVE_FILES}feature_selected_data.csv')
