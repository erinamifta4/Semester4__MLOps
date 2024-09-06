import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import math

import math

def kolmogorov_smirnov(y_pred, y_true):
    """
    Returns a results of Kolmogorov-smirnov test on goodness of fit using scipy.ks_2sample test.
    """
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic

    
def gini(y_true, y_pred, sample_weight=None):
    """
    Returns Gini coefficient (linear transformation of Area Under Curve with formula 2*AUC-1 )
    """
    return 2*roc_auc_score(y_true, y_pred, sample_weight=sample_weight)-1


def lift(y_true, y_pred, lift_perc = 10):
    """
    Returns Lift of prediction
    """
    cutoff=np.percentile(y_pred, lift_perc)       
    return y_true[y_pred<=cutoff].mean()/y_true.mean()

def iv(y_true, x):
    """
    Returns Information Value of a binned predictor
    """
    woe={}
    lin={}
    iv=0
    for v in np.unique(x):
        woe[v]=(1.*(len(x[(x==v)&(y_true==0)])+1)/(len(x[y_true==0])+1))/(1.*(len(x[(x==v)&(y_true==1)])+1)/(len(x[y_true==1])+1))
        woe[v]=math.log(woe[v])
        lin[v]=(1.*(len(x[(x==v)&(y_true==0)])+1)/(len(x[y_true==0])+1))-(1.*(len(x[(x==v)&(y_true==1)])+1)/(len(x[y_true==1])+1))
        iv=iv+woe[v]*lin[v]
    return iv