from scipy.stats.stats import mode
import statsmodels.graphics.gofplots as sm
import scipy.stats as sc
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pandas as pd
import numpy as np
import copy
from scipy.stats import t
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
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
n_split = len(X)
kfold_indexes = list(KFold(n_split, shuffle=True).split(X))


def getKfoldIndexes():
    return copy.deepcopy(kfold_indexes)


def evaluateModel(model, modelName, saveResultFiles=False):
    # Setup directory to save results
    # Creates the directory to save the results
    pathToSaveModelEval = None
    pathToSaveModelsDump = None
    if saveResultFiles:
        pathToSaveModelEval = f'../results/modelTrained/{modelName}'
        pathToSaveModelsDump = pathToSaveModelEval+'/trainedModels'
        Path(pathToSaveModelsDump).mkdir(parents=True, exist_ok=True)

    scores = cross_validate(model,
                            X,
                            y=np.ravel(Y),
                            cv=getKfoldIndexes(),
                            scoring={'mse': 'neg_mean_squared_error'},
                            return_estimator=True)
    yArray, yHatArray = computesYHat(
        scores, pathToSaveModelsDump, modelName=modelName)
    # generateGraphs(yArray, yHatArray, modelName,
    #               pathToSaveModelEval=pathToSaveModelEval)

    predNPArray = np.concatenate((yArray, yHatArray),
                                 axis=1)
    dfColumnsExport = ['Y', 'YHat']
    predDf = pd.DataFrame(predNPArray, columns=dfColumnsExport)
    predDf.to_csv(f'{pathToSaveModelEval}/predictions.csv',
                  sep=';',
                  decimal='.',
                  index=False)
    r2Result, mseResult, maeResult = computeMetrics(yArray, yHatArray)

    textToPlot = f'{modelName}\n' \
        f'R2:  {r2Result:7.4f}\n' \
        f'MSE: {mseResult:7.4f}\n' \
        f'MAE: {maeResult:7.4f}'
    scores['yHat'] = yHatArray
    scores['y'] = yArray
    scores['R2'] = r2Result
    scores['MSE'] = mseResult
    scores['MAE'] = maeResult
    scores['modelName'] = modelName
    print(textToPlot)
    return scores


def computesYHat(crosValidScores, pathToSaveModelsDump=None, modelName=None):
    # Uses all the estimators from LOOCV to make estimations
    resultList = crosValidScores['estimator']
    cross_val_indexes = getKfoldIndexes()
    listYhat = []
    listY = []
    # index of the for loop
    i = 0
    for est in resultList:
        x_temp = cross_val_indexes[i][1]
        if len(x_temp) > 0:
            ground_truth = Y[x_temp]
            x_temp = X[x_temp]
            pred = est.predict(x_temp)
            listYhat = listYhat + list(pred)
            listY = listY + list(ground_truth.reshape(1, -1)[0])
            dump(
                est, f'{pathToSaveModelsDump}/{modelName}_LOOCV_FOLD_{i}.joblib')
        else:
            print('Problem in estimation')
        i = i + 1
    listY = mmY.inverse_transform(np.asarray(listY).reshape(-1, 1))
    listYhat = mmY.inverse_transform(np.asarray(listYhat).reshape(-1, 1))
    return listY, listYhat


def computeMetrics(yArray, yHatArray):
    maeResult = mean_absolute_error(yArray, yHatArray)
    r2Result = r2_score(yArray, yHatArray)
    mseResult = mean_squared_error(yArray, yHatArray)
    return r2Result, mseResult, maeResult


def generateGraphs(yArray, yHatArray, modelName, pathToSaveModelEval=None):
    plt.clf()
    plt.style.use(['seaborn-ticks'])
    plt.figure(figsize=(6.5, 4.1))  # 4.75))

    # Plots the estimatives
    plt.plot(yArray, yHatArray, "o")
    # Plots a black line for comparation purpose
    _, xmax = plt.xlim()
    plt.plot([0, xmax], [0, xmax], 'k-')
    y0, ymax = plt.ylim()
    yDistanceY0_yMax = ymax - y0

    # Plots a linear fit between prediction and actual value
    linear = LinearRegression()
    linear.fit(yArray, yHatArray)
    plt.plot(yArray, linear.predict(yArray), '-', color='red')

    r2Result, mseResult, maeResult = computeMetrics(yArray, yHatArray)
    residualArray = yArray - yHatArray

    textToPlot = f'R2:  {r2Result:7.4f}\n' \
        f'MSE: {mseResult:7.4f}\n' \
        f'MAE: {maeResult:7.4f}'
    print(textToPlot)
    plt.text(0, ymax - yDistanceY0_yMax * 0.2,
             textToPlot,
             bbox=dict(facecolor='gray', alpha=0.5),
             family='monospace')
    # plt.title(modelName)
    plt.grid(True)
    plt.xlabel('Laboratory Determined Porosity [%]')
    plt.ylabel(modelName+' Estimated Porosity [%]')

    if pathToSaveModelEval:
        # Save Graph
        nameInGraph = modelName.split(' ')[0]
        plt.savefig(f'{pathToSaveModelEval}/scatterPlot{nameInGraph}.png',
                    bbox_inches='tight', pad_inches=0.01)
        # Save file metrics
        with open(pathToSaveModelEval+'/metrics.txt', mode='w') as f:
            f.write(f'R2: {r2Result}\n')
            f.write(f'MAE: {maeResult}\n')
            f.write(f'MSE: {mseResult}\n')
            f.write(f'Residuals: {residualArray}\n')
            f.write(f'Y: {yArray}\n')
            f.write(f'YHat: {yHatArray}\n')
    plt.show()
    residualPlot(modelName, residualArray, pathToSave=pathToSaveModelEval)


def residualPlot(modelName,
                 residualList,
                 pathToSave=None):
    plt.clf()
    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={
                                            "height_ratios": (.15, .85)},
                                        figsize=(6.5, 4.1))
    # figsize=(10, 7))
    ax_box.set_xlim((-15, 15))
    ax_hist.set_xlim((-15, 15))
    ax_hist.set_ylim((0, 12))
    ax_hist.set_xlabel(f'{modelName} Porosity Estimation Residual')
    ax_hist.set_ylabel('Frequency')
    customBins = np.arange(-15.5, 15.5, 1)
    # maxValueTicks = max(np.histogram(residualList, bins=customBins)[0]) + 1
    ax_hist.set_yticks(np.arange(0, 13, 1))
    ax_hist.set_xticks(np.arange(-15, 16, 3))
    sns.boxplot(x=residualList, ax=ax_box)
    sns.histplot(data=residualList,
                 bins=customBins,
                 kde=False, ax=ax_hist, legend=False, edgecolor="k", linewidth=1)
    ax_box.set(yticks=[])
    # f.suptitle(f'{modelName} Residuals', fontsize=18)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    if pathToSave is not None:
        nameInGraph = modelName.split(' ')[0]
        plt.savefig(f'{pathToSave}/residualsPlot{nameInGraph}.png',
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()

# Grid Search for the best hyperparameters


def findBestHyperparamsGridSearch(gridParameters, modelName, model):
    cv = RepeatedKFold(
        n_splits=10, n_repeats=10, random_state=0
    )
    gsCV = GridSearchCV(model,
                        gridParameters,
                        cv=cv,
                        n_jobs=-1)
    gsCV.fit(X, Y)
    results_df = pd.DataFrame(gsCV.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = (
        results_df
        .set_index(results_df["params"].apply(
            lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
    )
    print(results_df[
        ['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
    ])

    print(
        f'Best {modelName}:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')


# Lasso
gridParameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                  'max_iter': [100, 1000, 10000, 100000]}
findBestHyperparamsGridSearch(gridParameters, 'Lasso Reg', Lasso())
# Ridge
gridParameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                  'max_iter': [100, 1000, 10000, 100000]}
findBestHyperparamsGridSearch(gridParameters, 'Ridge Reg', Ridge())

# ElasticNet

gridParameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                  'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  'max_iter': [100, 1000, 10000, 100000]}
findBestHyperparamsGridSearch(gridParameters, 'ElasticNet', ElasticNet())


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
findBestHyperparamsGridSearch(gridParameters, 'kNN', KNeighborsRegressor())

# SVR Model
gridParameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ['auto', 5, 1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf'],
                  'epsilon': [0.1, 0.01, 0.05]}
findBestHyperparamsGridSearch(gridParameters, 'SVR', SVR())

# RF

gridParameters = {'n_estimators': [10, 50, 100, 200, 500],
                  'criterion': ['mse', 'mae']}
findBestHyperparamsGridSearch(gridParameters, 'RF', RandomForestRegressor())

# sorted(sklearn.metrics.SCORERS.keys())
# MLP
gridParameters = {'hidden_layer_sizes': [(5, 5), (15, 10),
                                         (20, 15, 10),
                                         (20, 15, 15, 10)],
                  'activation': ['relu'],
                  'solver': ['adam'],
                  'max_iter': [500, 1000, 1250, 1600],
                  'alpha': [0.01, 0.001, 0.0001],
                  'learning_rate': ['constant', 'adaptive'],
                  'batch_size': [1, 2, 3],
                  'learning_rate_init': [0.01, 0.001],
                  'early_stopping': [False]
                  }
findBestHyperparamsGridSearch(gridParameters, 'MLP', MLPRegressor())

###################################
# Training and evaluation of models

# Linear Regression
linear = LinearRegression()
linearEval = evaluateModel(linear, 'Linear Reg', saveResultFiles=True)

# Ridge Regression
ridge = Ridge(alpha=0.1, max_iter=100)
ridgeEval = evaluateModel(ridge, 'Ridge Reg', saveResultFiles=True)

# Lasso Regression
lasso = Lasso(alpha=0.00025, max_iter=1000)
lassoEval = evaluateModel(lasso, 'Lasso Reg', saveResultFiles=True)

# ElasticNet
elasticNet = ElasticNet(alpha=0.00025, l1_ratio=1, max_iter=1000)
elasticNetEval = evaluateModel(elasticNet, 'ElasticNet', saveResultFiles=True)

important_coeficients = []
coef = []
for est in elasticNetEval['estimator']:
    vec = np.vectorize(lambda x: 0 if x == 0 else 1)
    print(vec(est.coef_))
    coef.append(est.coef_)
    important_coeficients.append(vec(est.coef_))
important_coef_np = np.asfarray(important_coeficients)
coef = np.asarray(coef)
important_columns = vec(important_coef_np.sum(axis=0)).nonzero()[0]
teste = coef[:, important_columns]

plt.boxplot(teste[:, :])
plt.show()
dataset.columns[important_columns]

# KNN Model Evaluation
knn = KNeighborsRegressor(n_neighbors=2, metric='minkowski')
knnEval = evaluateModel(knn, 'KNN', saveResultFiles=True)

# SVR Model Evaluation
svr = SVR(gamma=1, C=10, epsilon=0.01, kernel='rbf')
svrEval = evaluateModel(svr, 'SVR', saveResultFiles=True)

# Random Forest
forest = RandomForestRegressor(n_estimators=50, criterion='mae')
forestEval = evaluateModel(forest, 'RF', saveResultFiles=True)

# MLP Model Evaluation
mlp = MLPRegressor(max_iter=1600, hidden_layer_sizes=(20, 15, 15, 10),
                   activation='relu', alpha=0.01, learning_rate='constant',
                   learning_rate_init=0.001, batch_size=3, solver='adam')
mlpEval = evaluateModel(mlp, 'MLP', saveResultFiles=True)


# Compile all the predictions in the same CSV file

crossValIndexes = getKfoldIndexes()
crossValIndexes = list(map(lambda x: x[1][0], crossValIndexes))
wavelengthColumns = list(dataset.columns[:-1])

yHatTable = np.concatenate((X[crossValIndexes], linearEval['y'], linearEval['yHat'], ridgeEval['yHat'], lassoEval['yHat'],
                            lassoEval['yHat'], knnEval['yHat'], svrEval['yHat'], forestEval['yHat'], mlpEval['yHat']),
                           axis=1)
dfColumnsExport = wavelengthColumns + ['Y', 'Linear', 'Ridge', 'Lasso',
                                       'ElasticNet', 'kNN', 'SVR', 'RF', 'MLP']
yHatDf = pd.DataFrame(yHatTable, columns=dfColumnsExport)
yHatDf.to_csv('../results/modelTrained/completePredictions.csv',
              sep=';',
              decimal='.',
              index=False)

indexColumns = ['modelName', 'R2', 'MSE', 'MAE']
summaryDF = pd.DataFrame(
    np.asarray(list(map(lambda x: list(map(lambda index: x[index], indexColumns)),
                        [linearEval, ridgeEval, lassoEval, elasticNetEval,
                        knnEval, svrEval, forestEval, mlpEval]))),
    columns=indexColumns)
summaryDF.to_csv('../results/modelTrained/summary.csv',
                 sep=';',
                 decimal='.',
                 index=False)


def plotAllGraphs():
    models = ['Linear Reg', 'Lasso Reg', 'Ridge Reg',
              'ElasticNet', 'KNN', 'SVR', 'RF', 'MLP']
    for modelName in models:
        pathToModelData = f'../results/modelTrained/{modelName}'
        pathToPredictionFile = f'{pathToModelData}/predictions.csv'
        predictionData = pd.read_csv(
            pathToPredictionFile, decimal='.', sep=';')
        yArray = predictionData['Y'].values.reshape(-1, 1)
        yHatArray = predictionData['YHat'].values.reshape(-1, 1)
        generateGraphs(yArray, yHatArray, modelName,
                       pathToSaveModelEval=pathToModelData)


plotAllGraphs()
