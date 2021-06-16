import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import re


def plotCorrelationMatrix(df, annot=True):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=annot)


dataset = pd.read_csv(f'../data/data.csv',
                      sep=';',
                      decimal=',')

dataset = dataset.drop(['seq', 'sample', 'place'], axis=1)

# wavelengths selected by visual inspection:
# 380, 405, 485, 670, 970, 1005, 1410,
# 1460, 1900, 1930, 2150, 2220, 2325


# def isVisuallySelectedFeature(columnName):
#    visualSelection = ['38.', '40.', '48.', '67.', '97.', '100.', '141.',
#                       '146.', '190.', '193.', '215.', '222.', '232.']

#    return sum(list(map(lambda x: 0 if re.search('^' + x, columnName) is None else 1,
#                        visualSelection))) >= 1
# columnsSelected = list(filter(isVisuallySelectedFeature, dataset.columns))

visualSelection = ['382', '405.4', '486.3', '670', '970.3', '1005.4', '1412.8',
                   '1461.8', '1900.9', '1932.2', '2151.3', '2222.1', '2324.3']

POROSITY_COLUMN_NAME = dataset.columns[-1]
columnsInDataset = list(dataset.columns)
DIVISIONS_IN_SPECTRA = 20
WAVELENGTHS_PER_DIVISION = len(columnsInDataset) // DIVISIONS_IN_SPECTRA
correlationSelection = []
for i in range(DIVISIONS_IN_SPECTRA):
    lowerCut = i * WAVELENGTHS_PER_DIVISION
    upperCut = lowerCut + WAVELENGTHS_PER_DIVISION
    # print(lowerCut,upperCut,columnsInDataset[lowerCut:upperCut])
    datasetCorr = dataset[columnsInDataset[lowerCut:upperCut] +
                          [POROSITY_COLUMN_NAME]]
    corr = datasetCorr.corr()[POROSITY_COLUMN_NAME]
    # plotCorrelationMatrix(datasetCorr, annot=False)


#corr = dataset.corr()
#Todos = corr[corr[dataset.columns[-1]].pow(2).pow(0.5) > 0.8][dataset.columns[-1]]


dataset = dataset[['1842.3', '1932.2', '2278.7'] + [dataset.columns[-1]]]

plotCorrelationMatrix(dataset)

print(dataset.columns)


dataset = dataset[visualSelection + [dataset.columns[-1]]]

# dataset = dataset.drop(dataset.columns[:50], axis=1)
# dataset = dataset.drop(dataset.columns[-50:-1], axis=1)
