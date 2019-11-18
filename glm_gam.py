import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from typing import List
from collections import namedtuple
import matplotlib.pyplot as plt

namedtuple("Classifier", ["classifier", "smooth_names"])
class GlmGam(object):
    def __init__(self, smooth_names: List, dfs: List , alphas, family = sm.families.Binomial(), degree = 3, **kwargs):
        self._smooth_names = smooth_names
        self._alphas = alphas
        self._dfs = dfs
        self._family = family
        self._res_bs = None
        self._normalization_methods = None
        self._lower_bound = kwargs.get("lower_bound", None)
        self._upper_bound = kwargs.get("upper_bound", None)
        self._degree = degree
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(self._smooth_names) == 0:
            bs = None
        else:
            X_spline = X[self._smooth_names]
            bs = BSplines(X_spline, df = self._dfs, 
                          degree = [self._degree] * len(self._smooth_names), 
                          knot_kwds = [{
                              "lower_bound": None if self.lower_bound is None else self.lower_bound[i], 
                              "upper_bound": None if self.upper_bound is None else self.upper_bound[i]
                          } for i in range(len(self._smooth_names))])
        self._gam_bs = GLMGam(y, X.iloc[:, ~ X.columns.isin(self._smooth_names)], 
                              smoother = bs, alpha = self._alphas, family = self._family)
        self._res_bs = self._gam_bs.fit()
        return self
    
    
    @property
    def lower_bound(self):
        return self._lower_bound
    
    
    @lower_bound.setter
    def lower_bound(self, val):
        self._lower_bound = val
        

    @property
    def upper_bound(self):
        return self._upper_bound
    
    
    @upper_bound.setter
    def upper_bound(self, val):
        self._upper_bound = val
        
    @property
    def smooth_names(self):
        return self._smooth_names
    
    
    def predict(self, X: pd.DataFrame):
        if self._res_bs is None:
            raise ValueError("Model is not fitted, result is not available!")
        X_smooth = X[self._smooth_names]
        X = X.iloc[:, ~ X.columns.isin(self._smooth_names)]
        return self._res_bs.predict(X.values, X_smooth.values)
    
    def _plot_partial(self, ind, cpr = True, ax = None, **kwargs):
        return self._res_bs.plot_partial(ind, cpr = cpr, ax = ax, **kwargs)
    
    def plot_partial(self, cpr = True, axes = None, r = None, c = None, **kwargs):
        ax_num = len(self._smooth_names)
        if axes is None:
            if c is None and r is None:
                r, c = 1, ax_num
            else:
                r, c = (r, ax_num//r) if c is None else (ax_num//c, c)
            _, axes = plt.subplots(r, c, figsize = (5 * c, 5 * r))
            axes = axes.reshape(r, c)
        for i in range(r):
            for j in range(c):
                self._plot_partial(i * c + j, cpr = cpr, ax = axes[i, j], **kwargs)
        return axes