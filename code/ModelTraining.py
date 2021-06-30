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
upper_scores = None


def getKfoldIndexes():
    return copy.deepcopy(kfold_indexes)


n_split = len(X)
kfold_indexes = list(KFold(n_split, shuffle=True).split(X))


def evaluateModel(model, modelName):
    global upper_scores
    scores = cross_validate(model,
                            X,
                            y=np.ravel(Y),
                            cv=getKfoldIndexes(),
                            scoring={'mse': 'neg_mean_squared_error'},
                            return_estimator=True)
    upper_scores = scores
    print(modelName, 'MSE', scores['test_mse'])
    # print('R2:',scores['test_r2'])
    generateGraphs(scores, modelName)

    return scores


def generateGraphs(crosValidScores, modelName):
    resultList = crosValidScores['estimator']
    # Uses all the estimators from LOOCV to make estimations
    varia = 0
    cross_val_indexes = getKfoldIndexes()
    plt.style.use(['seaborn-ticks'])
    listYhat = []
    listY = []
    # Creates the directory to save the results
    pathToSaveModelEval = f'../results/modelTrained/{modelName}'
    pathToSaveModelsDump = pathToSaveModelEval+'/trainedModels'
    Path(pathToSaveModelsDump).mkdir(parents=True, exist_ok=True)
    plt.clf()

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
    #

    # Scatter plot the estimations and the ground truth values
    plt.plot(listY, listYhat, "o")
    plt.plot([0, 1], [0, 1], 'k-')
    linear = LinearRegression()

    yArray = np.asarray(listY).reshape(len(listY), 1)
    yHatArray = np.asarray(listYhat).reshape(len(listYhat), 1)

    residualArray = yArray - yHatArray

    linear.fit(yArray, yHatArray)
    plt.plot(yArray, linear.predict(yArray), '-', color='red')
    plt.xlabel('Laboratory Determined Porosity')
    plt.ylabel(modelName+' Estimated Porosity')

    maeResult = mean_absolute_error(listY, listYhat)
    r2Result = r2_score(listY, listYhat)
    mseResult = mean_squared_error(listY, listYhat)

    print(f'R2: {r2Result}')
    print(f'MAE: {maeResult}')
    print(f'MSE: {mseResult}')
    print(f'Residuals: {residualArray}')

    #print('std:', statistics.stdev(meanSquaredErrorsList))
    plt.text(0, 0.85, f'R2: {round(r2Result, 4)}\nMSE: {round(mseResult, 6)}\nMAE: {round(maeResult,4)}',
             bbox=dict(facecolor='gray', alpha=0.5))
    plt.title(modelName)
    plt.grid(True)

    plt.savefig(pathToSaveModelEval+'/scatterPlot.png')

    with open(pathToSaveModelEval+'/metrics.txt', mode='w') as f:
        f.write(f'R2: {r2Result}\n')
        f.write(f'MAE: {maeResult}\n')
        f.write(f'MSE: {mseResult}\n')
        f.write(f'Residuals: {residualArray}\n')
        f.write(f'Y: {yArray}\n')
        f.write(f'YHat: {yHatArray}\n')
    residualPlot(modelName, residualArray, pathToSave=pathToSaveModelEval)


def residualPlot(modelName,
                 residualList,
                 pathToSave=None,
                 binsUse=10):
    plt.clf()
    plt.hist(residualList, bins=binsUse, density=False, edgecolor='black')
    titleGraph = f'{modelName} Residuals'
    plt.title(titleGraph)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid()
    plt.xlim((-1, 1))
    if pathToSave is not None:
        plt.savefig(pathToSave+'/residualsPlot.png')
    else:
        plt.show()

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
gsCV = GridSearchCV(MLPRegressor(max_iter=5000),
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
evaluateModel(linear, 'Linear Reg')

# Ridge Regression
ridge = Ridge(alpha=0.001, max_iter=100)
evaluateModel(ridge, 'Ridge Reg')

# Lasso Regression
lasso = Lasso(alpha=0.00025, max_iter=1000)
evaluateModel(lasso, 'Lasso Reg')

# KNN Model Evaluation
knn = KNeighborsRegressor(n_neighbors=1, metric='minkowski')
evaluateModel(knn, 'KNN')

# SVR Model Evaluation
svr = SVR(gamma=1, C=10, epsilon=0.05, kernel='rbf')
evaluateModel(svr, 'SVR')

# Random Forest
forest = RandomForestRegressor(n_estimators=100, criterion='mae')
evaluateModel(forest, 'RF')

# MLP Model Evaluation
mlp = MLPRegressor(max_iter=5000, hidden_layer_sizes=(20, 15, 15, 5),
                   activation='relu', alpha=0.0005, learning_rate='constant',
                   batch_size=1, solver='adam')
evaluateModel(mlp, 'MLP')
