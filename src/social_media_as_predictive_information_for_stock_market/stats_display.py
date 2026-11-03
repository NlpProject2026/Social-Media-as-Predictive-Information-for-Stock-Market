import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def myhist(df, column, ax, x_unit, y_unit="absolute frequency", bins=20, log=False):

    if log==True:
        data = np.log10(df[[column]])
        data.hist(bins=bins, ax=ax)
        ax.set_title(fr"Distribution of $Y_{{{column}}}$ in log scale")
        ax.set_xlabel(fr"log({x_unit})")
    else :
        data = df[[column]]
        data.hist(bins=bins, ax=ax)
        ax.set_title(fr"Distribution of $Y_{{{column}}}$")
        ax.set_xlabel(x_unit) 

    Q1 = data[column].quantile(0.25)
    Q2 = data[column].median()
    Q3 = data[column].quantile(0.75)
    skewness = data[column].skew()
    
    ax.axvline(Q1, color='k', linestyle='--', linewidth=3, label=f"first quartile Q1 = {Q1:.2f}")
    ax.axvline(Q2, color='b', linestyle='-.', linewidth=3, label=f"median = {Q2:.2f}")
    ax.axvline(Q3, color='k', linestyle=':', linewidth=3, label=f"third quartile Q3 = {Q3:.2f}")
    ax.text(0.5, 0.5, f"Skewness : {skewness:.2f}", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round, pad=0.5", facecolor='w', alpha=0.8))
    mode_idx = np.argmax([patch.get_height() for patch in ax.patches]) 
    ax.patches[mode_idx].set_facecolor("r")
    ax.set_ylabel(y_unit)
    ax.legend()

def boxplot_classification(data, intervals, column, column_class, ax, hue=None):
    sns.boxplot(x=column_class, y=column, hue=hue, data=data, ax=ax)
    labels = [f"]{interval[0]}, {interval[1]}]" for interval in intervals]
    ax.set_xticks(ax.get_xticks(), labels=labels, rotation=45, ha="center")