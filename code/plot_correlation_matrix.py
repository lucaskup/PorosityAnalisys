import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from correlation_matrix import plot_correlation_matrix


dataset = pd.read_csv(f'../data/data.csv',
                      sep=';',
                      decimal=',')

# Plots the correlation matrix as a heatmap of all the features
POROSITY_COLUMN_NAME = dataset.columns[-1]
DATASET_COLUMNS = list(dataset.columns)
DATASET_COLUMNS = DATASET_COLUMNS[53:-51] + [POROSITY_COLUMN_NAME]

filePathAndName = f'../results/featureSelection/CorrMatrix_Wav_Complete.png'
plot_correlation_matrix(dataset[DATASET_COLUMNS[:-2]],
                        annotate_cells=False,
                        file_name=filePathAndName)

# Plots the correlation graph between porosity and wavelengths
partial_correlation_matrix = dataset[DATASET_COLUMNS].corr()
x_graph_labels = np.asarray(DATASET_COLUMNS[:-1])
x_graph_values = list(range(len(x_graph_labels)))
y_graph = partial_correlation_matrix['Porosity (%)'].values[:-1]
plt.style.use(['seaborn-ticks'])
plt.plot(x_graph_values, y_graph)
plt.grid(True)
plt.xlabel('Wavelength [nm]')
ticks = [0, 153, 306, 459, 612, 765, 917]
plt.xticks(ticks=ticks, labels=x_graph_labels[ticks])
plt.ylabel('Correlation to Porosity [%]')
plt.savefig('../results/featureSelection/CorrelationPlot.png',
            dpi=450)
