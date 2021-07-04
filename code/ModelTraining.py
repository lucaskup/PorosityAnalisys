from os import sep
import statsmodels.graphics.gofplots as sm
import scipy.stats as sc
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import statistics

from pathlib import Path
import pandas as pd
import numpy as np
import copy

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_validate

from joblib import dump, load

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400


# Import the data
dataset = pd.read_csv('../results/featureSelection/featureSelectedData.csv',
                      index_col=0)

X = dataset.values[:, :-1].astype(np.float64)
Y = dataset['Porosity (%)'].values.astype(np.float64)

mmY = MinMaxScaler()

Y = mmY.fit_transform(Y.reshape(-1, 1)).ravel()

# Auxiliary Functions


def getKfoldIndexes():
    return copy.deepcopy(kfold_indexes)


n_split = len(X)
kfold_indexes = list(KFold(n_split, shuffle=True).split(X))


def evaluateModel(model, modelName):
    scores = cross_validate(model,
                            X,
                            y=np.ravel(Y),
                            cv=getKfoldIndexes(),
                            scoring={'mse': 'neg_mean_squared_error'},
                            return_estimator=True)
    print(modelName, 'MSE', scores['test_mse'])
    # print('R2:',scores['test_r2'])
    generateGraphs(scores, modelName)

    return scores


def computesYHat(crosValidScores, pathToSaveModelsDump, modelName):
    resultList = crosValidScores['estimator']
    # Uses all the estimators from LOOCV to make estimations
    varia = 0
    cross_val_indexes = getKfoldIndexes()
    listYhat = []
    listY = []
    for est in resultList:
        x_temp = cross_val_indexes[varia][1]
        if len(x_temp) > 0:
            ground_truth = Y[x_temp]
            x_temp = X[x_temp]
            pred = est.predict(x_temp)
            listYhat = listYhat + list(pred)
            listY = listY + list(ground_truth.reshape(1, -1)[0])
            dump(
                est, f'{pathToSaveModelsDump}/{modelName}_LOOCV_FOLD_{varia}.joblib')
        else:
            print('Problem in estimation')
        varia = varia + 1
    # print(listY)

    listY = mmY.inverse_transform(np.asarray(listY).reshape(-1, 1))
    listYhat = mmY.inverse_transform(np.asarray(listYhat).reshape(-1, 1))
    return listY, listYhat


def generateGraphs(crosValidScores, modelName):

    plt.clf()
    plt.style.use(['seaborn-ticks'])
    plt.figure(figsize=(7.5, 4.75))
    # Creates the directory to save the results
    pathToSaveModelEval = f'../results/modelTrained/{modelName}'
    pathToSaveModelsDump = pathToSaveModelEval+'/trainedModels'
    Path(pathToSaveModelsDump).mkdir(parents=True, exist_ok=True)

    yArray, yHatArray = computesYHat(
        crosValidScores, pathToSaveModelsDump, modelName)

    # Scatter plot the estimations and the ground truth values

    plt.plot(yArray, yHatArray, "o")
    maxXAxis = max(yArray)
    maxYAxis = max(yHatArray)
    plt.plot([0, maxXAxis], [0, maxXAxis], 'k-')
    linear = LinearRegression()

    #yArray = np.asarray(listY).reshape(len(listY), 1)
    #yHatArray = np.asarray(listYhat).reshape(len(listYhat), 1)

    residualArray = yArray - yHatArray

    linear.fit(yArray, yHatArray)
    plt.plot(yArray, linear.predict(yArray), '-', color='red')
    plt.xlabel('Laboratory Determined Porosity [%]')
    plt.ylabel(modelName+' Estimated Porosity [%]')

    maeResult = mean_absolute_error(yArray, yHatArray)
    r2Result = r2_score(yArray, yHatArray)
    mseResult = mean_squared_error(yArray, yHatArray)

    print(f'R2: {r2Result}')
    print(f'MAE: {maeResult}')
    print(f'MSE: {mseResult}')
    print(f'Residuals: {residualArray}')

    #print('std:', statistics.stdev(meanSquaredErrorsList))
    plt.text(0, maxYAxis - 4, f'R2: {round(r2Result, 4)}\nMSE: {round(mseResult, 6)}\nMAE: {round(maeResult,4)}',
             bbox=dict(facecolor='gray', alpha=0.5))
    plt.title(modelName)
    plt.grid(True)

    plt.savefig(pathToSaveModelEval+'/scatterPlot.png')
    plt.show()

    with open(pathToSaveModelEval+'/metrics.txt', mode='w') as f:
        f.write(f'R2: {r2Result}\n')
        f.write(f'MAE: {maeResult}\n')
        f.write(f'MSE: {mseResult}\n')
        f.write(f'Residuals: {residualArray}\n')
        f.write(f'Y: {yArray}\n')
        f.write(f'YHat: {yHatArray}\n')
    residualPlot(modelName, residualArray, pathToSave=pathToSaveModelEval)
    crosValidScores['yHat'] = yHatArray
    crosValidScores['y'] = yArray


def residualPlot(modelName,
                 residualList,
                 pathToSave=None):
    plt.clf()
    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={
                                            "height_ratios": (.15, .85)},
                                        figsize=(10, 7))

    ax_box.set_xlim((-25, 25))
    ax_hist.set_xlim((-25, 25))
    ax_hist.set_xlabel('Residual')
    ax_hist.set_ylabel('Frequency')
    customBins = np.arange(-25.5, 25.5, 1)
    maxValueTicks = max(np.histogram(residualList, bins=customBins)[0]) + 1
    ax_hist.set_yticks(np.arange(0, maxValueTicks, 1))
    ax_hist.set_xticks(np.arange(-25, 25, 5))
    sns.boxplot(residualList, ax=ax_box)
    sns.distplot(residualList,
                 bins=customBins,
                 kde=False, hist=True, ax=ax_hist,
                 hist_kws=dict(edgecolor="k", linewidth=1))

    ax_box.set(yticks=[])
    f.suptitle(f'{modelName} Residuals', fontsize=18)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    if pathToSave is not None:
        f.savefig(pathToSave+'/residualsPlot.png')
    f.show()


# Canonico
# plt.clf()
#plt.hist(residualList, bins=binsUse, density=False, edgecolor='black')
#titleGraph = f'{modelName} Residuals'
# plt.title(titleGraph)
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
# plt.grid()
#plt.xlim((-1, 1))
# if pathToSave is not None:
#    plt.savefig(pathToSave+'/residualsPlot.png')
# else:
#    plt.show()

# Grid Search for the best hyperparameters

# Lasso

gridParameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                  'max_iter': [100, 1000, 10000, 100000]}


gsCV = GridSearchCV(Lasso(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X, Y)

print(
    f'Best Lasso Regressor:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')


# Ridge

gridParameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                  'max_iter': [100, 1000, 10000, 100000]}


gsCV = GridSearchCV(Ridge(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X, Y)

print(
    f'Best Ridge Regressor:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')


# kNN
covParam = np.cov(X.astype(np.float32))
invCovParam = np.linalg.pinv(covParam)

gridParameters = [{'algorithm': ['auto'],
                  'metric': ['minkowski'],
                   'n_neighbors': [1, 2, 3, 4, 5]},
                  {'algorithm': ['brute'],
                   'metric': ['mahalanobis'],
                   'n_neighbors': [1, 2, 3, 4, 5],
                   'metric_params': [{'V': covParam,
                                      'VI': invCovParam}]}]
gsCV = GridSearchCV(KNeighborsRegressor(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X, Y)

print(
    f'Best kNN Classifier:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')

# SVR Model
gridParameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto', 5, 1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
gsCV = GridSearchCV(SVR(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X, Y)

print(
    f'Best SVM:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')

# RF

gridParameters = {'n_estimators': [10, 50, 100, 200, 500],
                  'criterion': ['mse', 'mae']}

gsCV = GridSearchCV(RandomForestRegressor(),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X, Y)

print(
    f'Best Random Forst Regressor:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')

# MLP

gridParameters = {'hidden_layer_sizes': [(10, 5), (10, 10), (10, 15),
                                         (15, 5), (15, 10), (15, 15),
                                         (5, 5, 5), (5, 5, 10), (5, 5, 15),
                                         (10, 10, 5), (10, 10, 10), (10, 10, 15),
                                         (10, 15, 5), (10, 15, 10), (10, 15, 15),
                                         (15, 5, 5), (15, 5, 10), (15, 5, 15),
                                         (15, 10, 5), (15, 10, 10), (15, 10, 15),
                                         (15, 15, 5), (15, 15, 10), (15, 15, 15),
                                         (20, 5, 5), (20, 15, 10), (20, 20, 15),
                                         (5, 15, 10, 5), (10, 15,
                                                          10, 10), (15, 15, 10, 15),
                                         (20, 15, 15, 5), (25, 15, 15, 10), (25, 15, 15, 15)],
                  'activation': ['logistic', 'relu'],
                  'solver': ['adam'],
                  'alpha': [0.05, 0.01, 0.001, 0.0005, 0.0001, 0.00001],
                  'learning_rate': ['constant', 'adaptive'],
                  'batch_size': [1, 2]
                  }
gsCV = GridSearchCV(MLPRegressor(max_iter=50000),
                    gridParameters,
                    cv=10,
                    n_jobs=-1)

gsCV.fit(X, Y)

print(
    f'Best MLP:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')


###################################
# Training and evaluation of models

# Linear Regression
linear = LinearRegression()
linearEval = evaluateModel(linear, 'Linear Reg')

# Ridge Regression
ridge = Ridge(alpha=0.001, max_iter=100)
ridgeEval = evaluateModel(ridge, 'Ridge Reg')

# Lasso Regression
lasso = Lasso(alpha=0.00025, max_iter=1000)
lassoEval = evaluateModel(lasso, 'Lasso Reg')


# KNN Model Evaluation
knn = KNeighborsRegressor(n_neighbors=1, metric='minkowski')
knnEval = evaluateModel(knn, 'KNN')

# SVR Model Evaluation
svr = SVR(gamma=1, C=10, epsilon=0.05, kernel='rbf')
svrEval = evaluateModel(svr, 'SVR')

# Random Forest
forest = RandomForestRegressor(n_estimators=100, criterion='mae')
forestEval = evaluateModel(forest, 'RF')

# MLP Model Evaluation
mlp = MLPRegressor(max_iter=50000, hidden_layer_sizes=(15, 15, 10, 15),
                   activation='relu', alpha=0.001, learning_rate='constant',
                   batch_size=1, solver='adam')
mlpEval = evaluateModel(mlp, 'MLP')

yHatTable = np.concatenate((linearEval['y'], linearEval['yHat'], ridgeEval['yHat'], lassoEval['yHat'],
                            knnEval['yHat'], svrEval['yHat'], forestEval['yHat'], mlpEval['yHat']),
                           axis=1)
yHatDf = pd.DataFrame(yHatTable, columns=['Y',
                      'Linear', 'Ridge', 'Lasso', 'kNN', 'SVR', 'RF', 'MLP'])
yHatDf.to_csv('../results/modelTrained/predictions.csv', sep=';', decimal='.')
