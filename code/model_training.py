import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import copy
from scipy.stats import t
from sklearn.model_selection import cross_validate, KFold, GridSearchCV, GroupKFold, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

# Import the data
dataset = pd.read_csv('../results/featureSelection/featureSelectedData.csv',
                      index_col=0)

X = dataset.values[:, 1:-1].astype(np.float64)
groups = dataset.values[:, 0].astype(np.str)
Y = dataset['Porosity (%)'].values.astype(np.float64)

mmY = MinMaxScaler()
Y = mmY.fit_transform(Y.reshape(-1, 1)).ravel()

# Auxiliary Functions
n_split = 10  # len(X)
kfold_indexes = list(GroupKFold(n_split).split(X, groups=groups))


def getKfoldIndexes():
    return copy.deepcopy(kfold_indexes)


def evaluate_model(model, model_name, save_results=False):
    '''
    Evaluates the model using LOOCV and creates a file with the results.

            Parameters:
                    model (sklearn.model): Sklearn model
                    model_name (int): Name of the algorithm
                    save_results (bool): save results to a file

            Returns:
                    scores (DataFrame): Results of LOOCV
    '''
    # Setup directory to save results
    # Creates the directory to save the results
    pathToSaveModelEval = None
    pathToSaveModelsDump = None
    if save_results:
        pathToSaveModelEval = f'../results/modelTrained/{model_name}'
        pathToSaveModelsDump = pathToSaveModelEval+'/trainedModels'
        Path(pathToSaveModelsDump).mkdir(parents=True, exist_ok=True)

    scores = cross_validate(model,
                            X,
                            y=np.ravel(Y),
                            cv=getKfoldIndexes(),
                            scoring={'mse': 'neg_mean_squared_error'},
                            return_estimator=True)
    yArray, yHatArray = computes_YHat(
        scores, pathToSaveModelsDump, model_name=model_name)

    predNPArray = np.concatenate((yArray, yHatArray),
                                 axis=1)
    dfColumnsExport = ['Y', 'YHat']
    predDf = pd.DataFrame(predNPArray, columns=dfColumnsExport)
    predDf.to_csv(f'{pathToSaveModelEval}/predictions.csv',
                  sep=';',
                  decimal='.',
                  index=False)
    r2Result, mseResult, maeResult = compute_metrics(yArray, yHatArray)

    textToPlot = f'{model_name}\n' \
        f'R2:  {r2Result:7.4f}\n' \
        f'MSE: {mseResult:7.4f}\n' \
        f'MAE: {maeResult:7.4f}'
    scores['yHat'] = yHatArray
    scores['y'] = yArray
    scores['R2'] = r2Result
    scores['MSE'] = mseResult
    scores['MAE'] = maeResult
    scores['modelName'] = model_name
    print(textToPlot)
    return scores


def computes_YHat(cv_scores,
                  path_to_save_models=None,
                  model_name=None):
    '''
     Uses all the estimators from LOOCV to make yHat estimations

            Parameters:
                    cv_scores (DataFrame): The return from a cross validation
                    path_to_save_models (String): Path to save model dump
                    model_name (String): Name of the model

            Returns:
                    y, y_hat (NumpyArray, NumpyArray): Ground Truth and Prediction
    '''

    resultList = cv_scores['estimator']
    cross_val_indexes = getKfoldIndexes()
    y_hat = []
    y = []
    # index of the for loop
    i = 0
    for est in resultList:
        x_temp = cross_val_indexes[i][1]
        if len(x_temp) > 0:
            ground_truth = Y[x_temp]
            x_temp = X[x_temp]
            pred = est.predict(x_temp)
            y_hat = y_hat + list(pred)
            y = y + list(ground_truth.reshape(1, -1)[0])
            dump(
                est, f'{path_to_save_models}/{model_name}_LOOCV_FOLD_{i}.joblib')
        else:
            print('Problem in estimation')
        i = i + 1
    y = mmY.inverse_transform(
        np.asarray(y).reshape(-1, 1))
    y_hat = mmY.inverse_transform(np.asarray(y_hat).reshape(-1, 1))
    return y, y_hat


def compute_metrics(y_array, y_hat_array):
    '''
    Returns metrics for the estimations passed as arguments.

            Parameters:
                    y_array (NumpyArray): Ground Truth
                    y_hat_array (NumpyArray): Model Estimations

            Returns:
                    (mae, r2, mse) (float, float, float): Metrics calculated
    '''
    mae = mean_absolute_error(y_array, y_hat_array)
    r2 = r2_score(y_array, y_hat_array)
    mse = mean_squared_error(y_array, y_hat_array)
    return r2, mse, mae


def create_graphs(y_array, y_hat_array,
                  model_name, path_save_evaluation=None):
    '''
    Creates scatter and residual plot of predictions passed in the
    first two parameters.

            Parameters:
                    y_array (NumpyArray): Ground Truth value
                    y_hat_array (NumpyArray): Estimated Values
                    model_name (String): Name of the models
                    path_save_evaluation (String): Path to save graphs and
                    metrics

            Returns:
                    None
    '''
    plt.clf()
    plt.style.use(['seaborn-ticks'])
    plt.figure(figsize=(6.5, 4.1))  # 4.75))

    # Plots the estimatives
    plt.plot(y_array, y_hat_array, "o")
    # Plots a black line for comparation purpose
    _, xmax = plt.xlim()
    plt.plot([0, xmax], [0, xmax], 'k-')
    y0, ymax = plt.ylim()
    yDistanceY0_yMax = ymax - y0

    # Plots a linear fit between prediction and actual value
    linear = LinearRegression()
    linear.fit(y_array, y_hat_array)
    plt.plot(y_array, linear.predict(y_array), '-', color='red')

    r2, mse, mae = compute_metrics(y_array, y_hat_array)
    residual_array = y_array - y_hat_array

    text_to_plot = f'R2:  {r2:7.4f}\n' \
        f'MSE: {mse:7.4f}\n' \
        f'MAE: {mae:7.4f}'
    print(text_to_plot)
    plt.text(0, ymax - yDistanceY0_yMax * 0.2,
             text_to_plot,
             bbox=dict(facecolor='gray', alpha=0.5),
             family='monospace')
    # plt.title(modelName)
    plt.grid(True)
    plt.xlabel('Laboratory Determined Porosity [%]')
    plt.ylabel(model_name+' Estimated Porosity [%]')

    if path_save_evaluation:
        # Save Graph
        name_in_graph = model_name.split(' ')[0]
        plt.savefig(f'{path_save_evaluation}/scatterPlot{name_in_graph}.png',
                    bbox_inches='tight', pad_inches=0.01)
        # Save file metrics
        with open(f'{path_save_evaluation}/metrics.txt',
                  mode='w') as f:
            f.write(f'R2: {r2}\n')
            f.write(f'MAE: {mae}\n')
            f.write(f'MSE: {mse}\n')
            f.write(f'Residuals: {residual_array}\n')
            f.write(f'Y: {y_array}\n')
            f.write(f'YHat: {y_hat_array}\n')
    plt.show()
    create_residual_plot(model_name, residual_array,
                         path_to_save=path_save_evaluation)


def create_residual_plot(model_name,
                         residual_list,
                         path_to_save=None):
    '''
    Creates the residual plot histogram.

            Parameters:
                    model_name (String): Name of the model in the graph
                    residual_list (NumpyArray): Residuals of the estimation
                    path_to_save (String): Path to save the residuals graph

            Returns:
                    None
    '''
    plt.clf()
    sns.set(style="ticks")
    _, (ax_box, ax_hist) = plt.subplots(2,
                                        sharex=True,
                                        gridspec_kw={
                                            "height_ratios": (.15, .85)},
                                        figsize=(6.5, 4.1))
    ax_box.set_xlim((-15, 15))
    ax_hist.set_xlim((-15, 15))
    ax_hist.set_ylim((0, 13))
    ax_hist.set_xlabel(f'{model_name} Porosity Estimation Residual')
    ax_hist.set_ylabel('Frequency')
    customBins = np.arange(-15.5, 15.5, 1)
    ax_hist.set_yticks(np.arange(0, 14, 1))
    ax_hist.set_xticks(np.arange(-15, 16, 3))
    sns.boxplot(x=residual_list, ax=ax_box)
    sns.histplot(data=residual_list,
                 bins=customBins,
                 kde=False, ax=ax_hist, legend=False, edgecolor="k", linewidth=1)
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    if path_to_save is not None:
        name_in_graph = model_name.split(' ')[0]
        plt.savefig(f'{path_to_save}/residualsPlot{name_in_graph}.png',
                    bbox_inches='tight', pad_inches=0.01)
    plt.show()


def grid_search_hyperparameters(grid_parameters, model_name, model, save_results=False):
    '''
    Does a 10 repetition 10-fold cross validation grid search to
    select the best model in the parameter grid

            Parameters:
                    grid_parameters (Dictionary): Grid parameters to use
                    in the model search
                    model_name (String): Name of the model in the graph
                    model (sklearn.model): Algorithm to use to train
                    the model
                    save_results (bool): Save LOOCV results to a file

            Returns:
                    best_params (Dictionary): Best parameters
    '''
    cv = RepeatedKFold(
        n_splits=10, n_repeats=10, random_state=0
    )
    gsCV = GridSearchCV(model,
                        grid_parameters,
                        cv=cv,
                        n_jobs=-1,
                        scoring='neg_mean_squared_error')
    gsCV.fit(X, Y)
    results_df = pd.DataFrame(gsCV.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = (
        results_df
        .set_index(results_df["params"].apply(
            lambda x: "_".join(f'{key}:{val}' for key, val in x.items()))
        )
        .rename_axis('model')
    )
    print(results_df[
        ['rank_test_score', 'mean_test_score', 'std_test_score']
    ])
    if save_results:
        results_df.drop('params',
                        axis=1).to_csv(f'../results/modelTrained/{model_name}/GridSearchCV.csv',
                                       decimal='.',
                                       sep=';')

    print(
        f'Best {model_name}:\n   Score > {gsCV.best_score_}\n   Params > {gsCV.best_params_}')
    return gsCV.best_params_


# Lasso
grid_parameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                   'max_iter': [100, 1000, 10000, 100000]}
grid_search_hyperparameters(grid_parameters,
                            'Lasso Reg',
                            Lasso(),
                            save_results=True)
# Ridge
grid_parameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                   'max_iter': [100, 1000, 10000, 100000]}
grid_search_hyperparameters(grid_parameters,
                            'Ridge Reg',
                            Ridge(),
                            save_results=True)

# ElasticNet

grid_parameters = {'alpha': [0.1, 0.01, 0.001, 0.0005, 0.00025, 0.0001, 0.00005],
                   'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   'max_iter': [100, 1000, 10000, 100000]}
grid_search_hyperparameters(grid_parameters,
                            'ElasticNet',
                            ElasticNet(),
                            save_results=True)


# kNN
covParam = np.cov(X.astype(np.float32))
invCovParam = np.linalg.pinv(covParam)
grid_parameters = [{'algorithm': ['auto'],
                    'metric': ['minkowski'],
                   'n_neighbors': [1, 2, 3, 4, 5]},
                   {'algorithm': ['brute'],
                   'metric': ['mahalanobis'],
                    'n_neighbors': [1, 2, 3, 4, 5],
                    'metric_params': [{'V': covParam,
                                      'VI': invCovParam}]}]
grid_search_hyperparameters(grid_parameters,
                            'KNN',
                            KNeighborsRegressor(),
                            save_results=True)

# SVR Model
grid_parameters = {'C': [0.1, 1, 10, 50],
                   'gamma': ['auto', 5, 1, 0.1, 0.01, 0.001, 0.0001],
                   'kernel': ['rbf', 'poly'],
                   'epsilon': [0.1, 0.01, 0.05]}
grid_search_hyperparameters(grid_parameters,
                            'SVR',
                            SVR(),
                            save_results=True)

# RF
grid_parameters = {'n_estimators': [10, 50, 100, 200, 500],
                   'criterion': ['mse', 'mae']}
grid_search_hyperparameters(grid_parameters,
                            'RF',
                            RandomForestRegressor(),
                            save_results=True)


# xGB

grid_parameters = {'learning_rate': [0.1, 0.05, 0.01],
                   'n_estimators': [50, 100, 200, 300, 400],
                   'ccp_alpha': [0]}
grid_search_hyperparameters(grid_parameters,
                            'GBoost',
                            GradientBoostingRegressor(),
                            save_results=True)


# sorted(sklearn.metrics.SCORERS.keys())
# MLP
grid_parameters = {'hidden_layer_sizes': [(16, 16), (32, 16),
                                          (32, 32, 32),
                                          (32, 32, 16, 16)],
                   'activation': ['relu'],
                   'solver': ['adam'],
                   'max_iter': [1000, 2000, 3000],
                   'alpha': [0.01, 0.001, 0.0001],
                   'learning_rate': ['constant', 'adaptive'],
                   'batch_size': [1, 2, 4],
                   'learning_rate_init': [0.01, 0.001],
                   'early_stopping': [True, False]
                   }
grid_search_hyperparameters(grid_parameters,
                            'MLP',
                            MLPRegressor(),
                            save_results=True)

###################################
# Training and evaluation of models

# Linear Regression
linear = LinearRegression()
linearEval = evaluate_model(linear, 'Linear Reg', save_results=True)

# Ridge Regression
ridge = Ridge(alpha=0.0001, max_iter=100)
ridgeEval = evaluate_model(ridge, 'Ridge Reg', save_results=True)

# Lasso Regression
lasso = Lasso(alpha=0.00005, max_iter=100)
lassoEval = evaluate_model(lasso, 'Lasso Reg', save_results=True)

# ElasticNet
elasticNet = ElasticNet(alpha=0.00005, l1_ratio=1, max_iter=100)
elasticNetEval = evaluate_model(elasticNet, 'ElasticNet', save_results=True)

'''
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
'''

# KNN Model Evaluation
knn = KNeighborsRegressor(n_neighbors=2,
                          metric='minkowski')
knnEval = evaluate_model(knn, 'KNN', save_results=True)

# SVR Model Evaluation
svr = SVR(gamma=5,
          C=10,
          epsilon=0.01,
          kernel='rbf')
svrEval = evaluate_model(svr, 'SVR', save_results=True)

# Random Forest
forest = RandomForestRegressor(n_estimators=500,
                               criterion='mae')
forestEval = evaluate_model(forest, 'RF', save_results=True)

# MLP Model Evaluation
mlp = MLPRegressor(max_iter=3000,
                   hidden_layer_sizes=(20, 15, 15, 10),
                   activation='relu',
                   alpha=0.001,
                   learning_rate='adaptive',
                   learning_rate_init=0.001,
                   batch_size=3,
                   solver='adam')
mlpEval = evaluate_model(mlp, 'MLP', save_results=True)


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


def plot_results():
    models = ['Linear Reg', 'Lasso Reg', 'Ridge Reg',
              'ElasticNet', 'KNN', 'SVR', 'RF', 'MLP']
    for model_name in models:
        path_model_data = f'../results/modelTrained/{model_name}'
        path_prediction_file = f'{path_model_data}/predictions.csv'
        df_prediction_data = pd.read_csv(
            path_prediction_file, decimal='.', sep=';')
        yArray = df_prediction_data['Y'].values.reshape(-1, 1)
        yHatArray = df_prediction_data['YHat'].values.reshape(-1, 1)
        create_graphs(yArray, yHatArray, model_name,
                      path_save_evaluation=path_model_data)


plot_results()
