import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_validation_curves(X, y, mod, param_name, param_range, cv = 5, scoring = "neg_mean_squared_error", n_jobs = 1, ax = None, **kwargs):
    from sklearn.model_selection import validation_curve
    train_scores, test_scores = validation_curve(
        mod, X, y, param_name=param_name, 
        param_range=param_range,
        cv=5, scoring=scoring, n_jobs=1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("validation curve")
    plt.xlabel(f"{param_name}")
    plt.ylabel("score")
    lw = kwargs.get("lw", 2)
    if ax is None:
        ax = plt
    ax.semilogx(param_range, train_scores_mean, label="training score",
                 color="darkorange", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    ax.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    ax.legend(loc="best")
    if ax is None:
        ax.show()
    else:
        return ax

def plot_lasso_path(X: np.ndarray, y: np.ndarray, ax = None):
    from sklearn.linear_model import lars_path
    _, _, coefs = lars_path(X, y, method='lasso', verbose=True)

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    
    if ax is None:
        plt.plot(xx, coefs.T)
        ymin, ymax = plt.ylim()
        plt.vlines(xx, ymin, ymax, linestyle='dashed')
        plt.xlabel('|coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.title('LASSO Path')
        plt.axis('tight')
        plt.show()
        return
    
    ax.plot(xx, coefs.T)
    ymin, ymax = ax.get_ylim()
    ax.vlines(xx, ymin, ymax, linestyle='dashed')
    ax.set_xlabel('|coef| / max|coef|')
    ax.set_ylabel('Coefficients')
    ax.set_title('LASSO Path')
    ax.axis('tight')
    return ax


def find_lar_coef(lmbda_prop, lmbda_cutoffs, coefs):
    lmbda = 0.2 * lmbda_cutoffs[-1]
    lower_ind = 0
    lmbda_cutoffs = list(lmbda_cutoffs[::-1])
    for i in range(len(lmbda_cutoffs)):
        if lmbda_cutoffs[i] > lmbda:
            break
        lower_ind += 1

    lower_lmbda, upper_lmbda = lmbda_cutoffs[lower_ind], lmbda_cutoffs[lower_ind + 1]

    lower_coef = coefs[:, lower_ind]
    upper_coef = coefs[:, lower_ind + 1]
    coef_for_lmbda = lower_coef * (upper_lmbda - lmbda) / (upper_lmbda - lower_lmbda) + \
    upper_coef * (lmbda - lower_lmbda) / (upper_lmbda - lower_lmbda)
    return coef_for_lmbda


def plot_roc_curve(clf, X: pd.DataFrame, y: pd.DataFrame, cv = 5, n_jobs = 1):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    cv = StratifiedKFold(n_splits = cv, random_state = 32)
    tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        mean = train.mean()
        std = train.std()
        f = lambda X: (X - mean) / std
        train, test = f(train), f(test)
        probas_ = clf.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
        fpr, tpr, threholds = roc_curve(y.iloc[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 1, alpha = 0.3, 
                 label = "ROC fold %d (AUC = %0.2f)" % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle = "--", lw = 2, color = "r", 
             label = "Chance", alpha = 0.8)
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color = 'b', 
             label = r'Mean ROC (AUC = %0.2f $\pm$%0.2f)' %(mean_auc, std_auc), 
             lw = 2, alpha = 0.8)
    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                     alpha = .2, label = r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating charateristic example')
    plt.legend(loc = "lower right")
    plt.show()
    

    
    
    
                     