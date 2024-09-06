import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown

import category_encoders.woe as ce
from sklearn.metrics import roc_auc_score

import math


# Memory usage optimization
def optimizer(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is: {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if str(col_type)[:3] == 'int' or str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0 and str(col_type)[:3] == 'int':
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)            
            elif str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# Data Split
def data_split(df,
               sample_sizes = [0.6,0.2,0.2],
               sample_names = ['train','valid','test'],
               seed=None):
    
    if len(sample_sizes) != len(sample_names) or sum(sample_sizes) != 1:
        print('wrong input, please adjust the size and name sample')      
    else:    
        np.random.seed(seed)
        l = len(df)
        p = np.random.permutation(df.index)
        train = int(sample_sizes[0] * l)
        valid = int(sample_sizes[1] * l) + train
        test = int(sample_sizes[1] * l)

        dtype = np.where(df.index.isin(p[:train]), sample_names[0],
                            np.where(df.index.isin(p[train:valid]), sample_names[1], sample_names[2]))  
        print(pd.value_counts(dtype))
    
        return dtype
    
# Outlier Detection with IQR
def outlier_thresholds(df, col):
    """ function to estimate outlier limit
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit


def cnt_outliers(df, cols_pred, plot=False):
    """ function to estimate number of outlier case on inputted predictor
    """
    col_names = []
    for col in cols_pred:
        low_lim, up_lim = outlier_thresholds(df, col)
        if df[(df[col] > up_lim) | (df[col] < low_lim)].any(axis=None):
            number_of_outliers = df[(df[col] > up_lim) | (df[col] < low_lim)].shape[0]
            print(col, "'s outliers count:", number_of_outliers)
            col_names.append(col)
            if plot:
                sns.boxplot(x=df[col])
                plt.show()
    return col_names

def replace_with_thresholds(df, col):
    """ function to remove outlier
    """
    low_lim, up_lim = outlier_thresholds(df, col)
    df.loc[(df[col] < low_limit), col] = low_limit
    df.loc[(df[col] > up_limit), col] = up_limit
    
# Categorical Transformation - WoE
def woe_transform(df,mask,cat_columns,col_target):
    """ function to transform the predictor to numerical based on weight of default estimation
    """
    woe = ce.WOEEncoder(drop_invariant=True)
    df_woe = woe.fit(df[mask][cat_columns],df[mask][col_target]).transform(df[cat_columns])
    return df_woe

# Cutoff estimation
def cutoff_df(df, col_score, col_target):
    """ function to create dataframe PD score vs default rate
    """
    # initialized
    df = df.sort_values(col_score)
    
    # find the cumulative count and cumulative mean (expected default rate)
    df['cumm_def'] = np.cumsum(df[col_target])
    tot_def = df[col_target].sum()
    df['cumm_def_rate'] = df['cumm_def'].apply(lambda x: x/tot_def)
    return df[[col_score, 'cumm_def_rate']]