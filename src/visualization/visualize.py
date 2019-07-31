import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def print_correlated_columns(df, columns, print_correlated_hard = False, print_correlated_slight=False):
    """
    Este código imprime: 
    - la matriz de correlación
    - la matriz de correlación de atributos levemente lineales (0.8>x>0.5)
    - la matriz de correlación de atributos fuertemente lineales (>0.8)"""
    corr_matrix = df[columns].corr()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    upper_triangle_transpose = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    upper = corr_matrix.where(upper_triangle)
    slight_lineal_cols =  [ 
        column for column in upper.columns 
        if any(0.8 > np.abs(upper[column])) and np.abs(any(upper[column]) >= 0.5)
    ]
    lineal_cols = [column for column in upper.columns if np.abs(any(upper[column]) >= 0.8)]
#     print("levemente lineales  ", slight_lineal_cols)
#     print("fuertemente lineales ", lineal_cols)
    if print_correlated_hard:
        return (
            upper
            .loc[:, lineal_cols]
            .sort_values(by=lineal_cols, ascending=False)
            .stack()
            .dropna()
            .reset_index()
            .loc[lambda df: df[0] >= .8]
        )
    elif print_correlated_slight:
        return (
            upper
            .loc[:, slight_lineal_cols]
            .sort_values(by=slight_lineal_cols, ascending=False)
            .stack()
            .dropna()
            .reset_index()
            .loc[lambda df: (0.8 > np.abs(df[0])) & (np.abs(df[0]) >= 0.5)]
        )
    else:
        return df[columns].corr()


def plot_importance_reg(reg, columns, title):
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
    
    top_n = 10
    feat_imp = pd.DataFrame({'importance':reg.feature_importances_})    
    feat_imp['feature']=columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    ax = feat_imp.plot.barh(title=title, figsize=(12,6))
    plt.xlabel('Feature Importance Score')
    plt.savefig("../reports/figures/05-01-{}-feature_importance.svg".format(title))
    plt.show()
    return ax

