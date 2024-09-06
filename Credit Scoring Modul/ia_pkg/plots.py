import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown

def stacked_plot(df,cat_columns,col_target):
    """ function to plot categorical predictor distribution to target
    """
    r = len(cat_columns)
    for c,num in zip(cat_columns,range(1,r+1)):
        display(Markdown('***'))
        display(Markdown('### {}'.format(c)))
        _ = df.groupby(c)[col_target].agg([lambda x:1-np.mean(x), np.mean])
        _.columns = ['Not default', 'Default']
        
        f,ax = plt.subplots(figsize=(10,5))
        _ = _.plot(kind='barh',stacked=True, ax=ax)
        _ = ax.set_xlim(xmin=0.5)
        _ = ax.set_xticks([i*0.1 for i in range(5,11)])
        vals= ax.get_xticks()
        _ = ax.set_xlabel('% total user')
        _ = ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)
        _ = plt.title(c)
        _ = plt.show()

def dist_plot(df,columns,col_target):
    """ function to plot numerical predictor distribution to target
    """    
    r = len(columns)
    for c, i in zip(columns,range(0,r)):
        display(Markdown('***'))
        display(Markdown('### {}'.format(c)))
        f, ax1 = plt.subplots(figsize=(8,2))
        _ = sns.kdeplot(df.loc[df[col_target] == 0,c],ax=ax1,label='Not default',color='blue',shade=True)
        
        ax2 = ax1.twinx()
        _ = sns.kdeplot(df.loc[df[col_target] == 1,c],ax=ax2,label='Default',color='orange',shade=True)
        _ = ax1.set_ylabel('Density', fontsize = 10)
        lgd = ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)
        lgd1 = ax2.legend(bbox_to_anchor=(1, 0.9), loc='upper right', borderaxespad=0.1)
        _ = plt.title(c)
        _ = plt.show()

        
def plot_score_linearity(df, col_target, col_score, bins=10, ax=None):
    """ function to plot linearity by decile
    """
    df = df.loc[df[col_score].notnull(), [col_score, col_target]].copy()
    df['bin'], labels = pd.qcut(df[col_score], bins, retbins=True, precision=3)
    # labels
    grouped = df.groupby('bin')[col_target].aggregate(['sum', 'count'])
    grouped['rate'] = grouped['sum'] / grouped['count']
    grouped.columns = ['cnt_default', 'cnt_total', 'default_rate']
    
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10,6))
    else:
        ax1 = ax
        
    _ = grouped.plot.bar(y='cnt_total', ax=ax1, legend=False)
    _ = grouped.plot.bar(y='cnt_default', color='r', ax=ax1, legend=False)
    _ = plt.setp(ax1.get_xticklabels(), rotation=70, horizontalalignment='right')

    ax2 = ax1.twinx()

    _ = grouped.plot(y='default_rate', ax=ax2, style='r-o', legend=False)
    _ = ax1.set_title(col_score)
    _ = ax1.set_ylabel('count')
    _ = ax2.set_ylabel('rate')
    lgd = ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.1)
    _ = plt.show()

def cutoff_plot(df, col_score, exp_def_rate = 0.5, ax=None):
    """ plot score vs default rate with PD score cutoff
    """
    df['diff'] = df['cumm_def_rate']-exp_def_rate
    df['diff_abs'] = df['diff'].abs()
    idx = np.argmin(np.array(df['diff_abs']))
    cutoff = df[col_score].values[idx]
    
    print('The {} cutoff to {:.2%} default rate estimation: {}'.format(col_score,exp_def_rate,cutoff))
    if ax is None:
        ax = plt.gca()

    ax.plot(df[col_score], df['cumm_def_rate'], color='b')
    ax.axvline(cutoff, color = 'r')
    ax.scatter(cutoff, exp_def_rate, marker='x', color='k')
    ax.set_xticks([i*0.1 for i in range(1,11)])
    ax.set_yticks([i*0.1 for i in range(1,11)])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    
    