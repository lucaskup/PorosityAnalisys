from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from correlation_matrix import plot_correlation_matrix


dataset = pd.read_csv(f'../data/data.csv',
                      sep=';',
                      decimal=',')

porosity_roncador = dataset[dataset['place']
                            == 'Cachoeira_do_Roncador']['Porosity (%)']
porosity_sal = dataset[dataset['place'] !=
                       'Cachoeira_do_Roncador']['Porosity (%)']


def calculate_stats(porosity_values):
    return (round(np.mean(porosity_values), 2),
            round(np.median(porosity_values), 2),
            round(np.min(porosity_values), 2),
            round(np.max(porosity_values), 2),
            round(skew(porosity_values, bias=False), 2),
            round(kurtosis(porosity_values, bias=False, fisher=True), 2))


mean_roncador, median_roncador, min_roncador, max_roncador, skew_roncador, kurt_roncador = calculate_stats(
    porosity_roncador)

print(f'Roncador Stats:\n'
      f'mean {mean_roncador}\n'
      f'median {median_roncador}\n'
      f'min {min_roncador}\n'
      f'max {max_roncador}\n'
      f'skew {skew_roncador}\n'
      f'kurt {kurt_roncador}')

mean_sal, median_sal, min_sal, max_sal, skew_sal, kurt_sal = calculate_stats(
    porosity_sal)

print(f'sal Stats:\n'
      f'mean {mean_sal}\n'
      f'median {median_sal}\n'
      f'min {min_sal}\n'
      f'max {max_sal}\n'
      f'skew {skew_sal}\n'
      f'kurt {kurt_sal}')


dataset = dataset.drop(['seq', 'sample', 'place'], axis=1)
visual_selection = ['422.4', '486.3', '670', '970.3', '1005.4', '1412.8',
                    '1461.8', '1900.9', '1932.2', '2151.3', '2222.1', '2324.3']

POROSITY_COLUMN_NAME = dataset.columns[-1]
DATASET_COLUMNS = list(dataset.columns)
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

    corr_above_075 = partial_correlation_matrix[partial_correlation_matrix[POROSITY_COLUMN_NAME
                                                                           ].abs() > 0.75][POROSITY_COLUMN_NAME]
    print('Best Ones:\n', corr_above_075)
    corr_above_075.to_csv(f'{filePathAndName}.csv')


correlation_matrix_selection = ['853.1', '940.5', '996.2',
                                '1842.3', '1932.2', '2057.2', '2166.7', '2352.6']
selected_features_all_methods = list(
    dict.fromkeys(visual_selection + correlation_matrix_selection))
selected_features_all_methods = list(
    sorted(selected_features_all_methods, key=lambda x: float(x)))
feature_selected_data = dataset[selected_features_all_methods +
                                [dataset.columns[-1]]]
plot_correlation_matrix(feature_selected_data,
                        file_name='../results/featureSelection/CorrMatrix_Selected_Wav_png')
print(feature_selected_data.columns)

feature_selected_data.to_csv(
    '../results/featureSelection/featureSelectedData.csv')
