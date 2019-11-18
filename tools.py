import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

def cal_benchmark_perf(X_train, y_train, X_test, y_test):    
    mod = sm.OLS(y_train, X_train)
    regr = mod.fit()
    y_test_hat = regr.predict(X_test)
    benchmark = mean_squared_error(y_test_hat, y_test)
    return benchmark
    

def cal_perf(X_train, y_train, X_test, y_test, param_grid, return_grid = True):
    benchmark_perf = cal_benchmark_perf(X_train, y_train, X_test, y_test)
    regr =  mod = Lasso(fit_intercept = True)
    grid = GridSearchCV(regr, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs = 2, verbose = 1)
    grid.fit(X_train, y_train)
    y_test_hat = grid.predict(X_test)
    mod_perf = mean_squared_error(y_test_hat, y_test) / benchmark_perf
    if return_grid:
        return mod_perf, grid
    return mod_perf


def load_boston(path = "./data_base", online = False):
    if online:
        boston = datasets.load_boston()
        data = pd.DataFrame(data= np.c_[boston['data'], boston['target']], 
                            columns= list(boston['feature_names']) + ['target'])
        return data, boston['DESCR']
    with open(f"{path}/boston_description.txt", "r") as f:
        desc = f.readlines()
    desc = "".join(desc)
    return pd.read_csv(f"{path}/boston.csv"), desc


def load_cancer(path = "./data_base", online = False):
    if online:
        cancer = datasets.load_breast_cancer()
        data = pd.DataFrame(data= np.c_[cancer['data'], cancer['target']], 
                            columns= list(cancer['feature_names']) + ['target'])
        return data, cancer['DESCR']
    with open(f"{path}/cancer_description.txt", "r") as f:
        desc = f.readlines()

    desc = "".join(desc)
    return pd.read_csv(f"{path}/cancer.csv"), desc


