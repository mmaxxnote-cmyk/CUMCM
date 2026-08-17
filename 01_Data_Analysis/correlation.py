import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def pearson_correlation(data):
    """
    Pearson相关系数矩阵
    """

    return data.corr(method="pearson")



def spearman_correlation(data):
    """
    Spearman相关系数矩阵
    """

    return data.corr(method="spearman")
