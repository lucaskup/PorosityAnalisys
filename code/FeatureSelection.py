import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_correlation_matrix(df, annotate_cells=True, file_name=None):
    '''
    Uses matplot lib to plot a correlation matrix between the columns of the dataframe.

            Parameters:
                    df (DataFrame): Data used to plot the correlation matrix
                    annotate_cells (bool): When true annotate the cell with the correlation
                    file_name (String): Path and filename to save the plot

            Returns:
                    correlation (DataFrame): Correlation DataFrame
    '''
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=annotate_cells)
    if file_name is not None:
        plt.savefig(file_name, dpi=400)
    return corr


dataset = pd.read_csv(f'../data/data.csv',
                      sep=';',
                      decimal=',')

dataset = dataset.drop(['seq', 'sample', 'place'], axis=1)
visual_selection = ['422.4', '486.3', '670', '970.3', '1005.4', '1412.8',
                    '1461.8', '1900.9', '1932.2', '2151.3', '2222.1', '2324.3']

POROSITY_COLUMN_NAME = dataset.columns[-1]
DATASET_COLUMNS = list(dataset.columns)
DIVISIONS_IN_SPECTRA = 20
WAVELENGTHS_PER_DIVISION = len(DATASET_COLUMNS) // DIVISIONS_IN_SPECTRA
for i in range(DIVISIONS_IN_SPECTRA):
    lower_index_cut = i * WAVELENGTHS_PER_DIVISION
    lower_column_name = DATASET_COLUMNS[lower_index_cut]
    upper_index_cut = lower_index_cut + WAVELENGTHS_PER_DIVISION
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
plot_correlation_matrix(feature_selected_data)
print(feature_selected_data.columns)

feature_selected_data.to_csv(
    '../results/featureSelection/featureSelectedData.csv')
