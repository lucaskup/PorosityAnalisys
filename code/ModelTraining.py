import sklearn
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


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : int
        Variance-corrected standard deviation of the set of differences.
    """
    n = n_train + n_test
    corrected_var = (
        np.var(differences, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def getKfoldIndexes():
    return copy.deepcopy(kfold_indexes)


def evaluateModel(model, modelName, saveResultFiles=False):
    scores = cross_validate(model,
                            X,
                            y=np.ravel(Y),
                            cv=getKfoldIndexes(),
                            scoring={'mse': 'neg_mean_squared_error'},
                            return_estimator=True)
    generateGraphs(scores, modelName, saveResultFiles=saveResultFiles)
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


def generateGraphs(crosValidScores, modelName, saveResultFiles=False):
    plt.clf()
    plt.style.use(['seaborn-ticks'])
    plt.figure(figsize=(7.5, 4.75))

    # Creates the directory to save the results
    pathToSaveModelEval = None
    pathToSaveModelsDump = None
    if saveResultFiles:
        pathToSaveModelEval = f'../results/modelTrained/{modelName}'
        pathToSaveModelsDump = pathToSaveModelEval+'/trainedModels'
        Path(pathToSaveModelsDump).mkdir(parents=True, exist_ok=True)

    # Plots the estimatives
    yArray, yHatArray = computesYHat(
        crosValidScores, pathToSaveModelsDump, modelName=modelName)
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

    maeResult = mean_absolute_error(yArray, yHatArray)
    r2Result = r2_score(yArray, yHatArray)
    mseResult = mean_squared_error(yArray, yHatArray)
    residualArray = yArray - yHatArray

    textToPlot = f'R2:  {r2Result:2.4f}\n' \
        f'MSE: {mseResult:2.4f}\n' \
        f'MAE: {maeResult:2.4f}'
    print(textToPlot)
    plt.text(0, ymax - yDistanceY0_yMax * 0.2,
             textToPlot,
             bbox=dict(facecolor='gray', alpha=0.5),
             family='monospace')
    plt.title(modelName)
    plt.grid(True)
    plt.xlabel('Laboratory Determined Porosity [%]')
    plt.ylabel(modelName+' Estimated Porosity [%]')

    if saveResultFiles:
        # Save Graph
        plt.savefig(pathToSaveModelEval+'/scatterPlot.png')
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
    sns.boxplot(x=residualList, ax=ax_box)
    sns.histplot(data=residualList,
                 bins=customBins,
                 kde=False, ax=ax_hist, legend=False, edgecolor="k", linewidth=1)
    ax_box.set(yticks=[])
    f.suptitle(f'{modelName} Residuals', fontsize=18)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    if pathToSave is not None:
        plt.savefig(pathToSave+'/residualsPlot.png')
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
gridParameters = {'hidden_layer_sizes': [(5, 5), (5, 10), (5, 20),
                                         (10, 5), (10, 10), (10, 20),
                                         (20, 5), (20, 10), (15, 15),
                                         (20, 15, 10), (20, 15, 15, 10)],
                  'activation': ['relu'],
                  'solver': ['adam'],
                  'max_iter': [50, 500, 1000, 1250, 2000, 5000],
                  'alpha': [0.01, 0.001, 0.0001],
                  'learning_rate': ['constant'],
                  'batch_size': [1, 2, 3],
                  'learning_rate_init': [0.01, 0.001, 0.0001],
                  'early_stopping': [True, False]
                  }

# gridParameters = {'hidden_layer_sizes': [(20, 15, 15, 10), (20, 15, 10), (20, 10)],
#                  'max_iter': [1000, 1250, 1500, 1750],
#                  'activation': ['relu'],
#                  'solver': ['adam'],
#                  'alpha': [0.001],
#                  'learning_rate': ['adaptive'],
#                  'batch_size': [2],
#                  'learning_rate_init': [0.001],
#                  'early_stopping': [False]
#                  }
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

# KNN Model Evaluation
knn = KNeighborsRegressor(n_neighbors=1, metric='minkowski')
knnEval = evaluateModel(knn, 'KNN', saveResultFiles=True)

# SVR Model Evaluation
svr = SVR(gamma=1, C=100, epsilon=0.01, kernel='rbf')
svrEval = evaluateModel(svr, 'SVR', saveResultFiles=True)

# Random Forest
forest = RandomForestRegressor(n_estimators=500, criterion='mae')
forestEval = evaluateModel(forest, 'RF', saveResultFiles=True)

# MLP Model Evaluation
mlp = MLPRegressor(max_iter=1250, hidden_layer_sizes=(20, 15, 15, 10),
                   activation='relu', alpha=0.001, learning_rate='adaptive',
                   learning_rate_init=0.001, batch_size=2, solver='adam')
mlpEval = evaluateModel(mlp, 'MLP', saveResultFiles=True)


# Compile all the predictions in the same CSV file

crossValIndexes = getKfoldIndexes()
crossValIndexes = list(map(lambda x: x[1][0], crossValIndexes))
wavelengthColumns = list(dataset.columns[:-1])

yHatTable = np.concatenate((X[crossValIndexes], linearEval['y'], linearEval['yHat'], ridgeEval['yHat'], lassoEval['yHat'],
                            knnEval['yHat'], svrEval['yHat'], forestEval['yHat'], mlpEval['yHat']),
                           axis=1)
dfColumnsExport = wavelengthColumns + ['Y', 'Linear', 'Ridge',
                                       'Lasso', 'kNN', 'SVR', 'RF', 'MLP']
yHatDf = pd.DataFrame(yHatTable, columns=dfColumnsExport)
yHatDf.to_csv('../results/modelTrained/predictions.csv',
              sep=';',
              decimal='.',
              index=False)


def yetToBeNamed(cvResults):

    results_df = pd.DataFrame(cvResults.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = (
        results_df
        .set_index(results_df["params"].apply(
            lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
    )
    results_df[
        ['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
    ]

    # create df of model scores ordered by perfomance
    model_scores = results_df.filter(regex=r'split\d*_test_score')
    cv = RepeatedKFold(
        n_splits=10, n_repeats=10, random_state=0
    )
    # plot 30 examples of dependency between cv fold and AUC scores
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False, palette='Set1', marker='o', alpha=.5, ax=ax
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Model Neg Mean Squared Error", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    plt.show()

    # print correlation of AUC scores across folds
    print(f"Correlation of models:\n {model_scores.transpose().corr()}")

    model_1_scores = model_scores.iloc[0].values  # scores of the best model
    # scores of the second-best model
    model_2_scores = model_scores.iloc[1].values

    differences = model_1_scores - model_2_scores

    n = differences.shape[0]  # number of test sets
    df = n - 1
    n_train = len(list(cv.split(X, Y))[0][0])
    n_test = len(list(cv.split(X, Y))[0][1])

    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    print(f"Corrected t-value: {t_stat:.3f}\n"
          f"Corrected p-value: {p_val:.3f}")

    t_stat_uncorrected = (
        np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
    )
    p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)

    print(f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
          f"Uncorrected p-value: {p_val_uncorrected:.3f}")
