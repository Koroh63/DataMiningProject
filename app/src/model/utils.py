import random
import math
import statistics
from scipy.stats import norm, qmc
import numpy as np
from sklearn import preprocessing, datasets
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris = load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])


def calcul_correlation(x, y):
    Sx = np.std(x);
    Sy = np.std(y);

    mx = np.mean(x);
    my = np.mean(y);

    res = 0

    for i in range(len(x)):
        res += (x[i] - mx) * (y[i] - my);

    return (res/len(x))/(Sx*Sy);

def calcul_all_correlation(data):
    dic = {}
    for i in data:
        for j in data:
            if i != j and i+" - "+j not in dic and j+" - "+i not in dic:
                dic[i+" - "+j] = calcul_correlation(data[i], data[j]);
    return dic

print(calcul_all_correlation(data1))

def calcul_all_interval_confiance(data, interval):
    dic = {}
    for i in data:
        for j in data:
            if i != j and i+" - "+j not in dic and j+" - "+i not in dic:
                res = calcul_correlation(data[i], data[j]);
                dic[i+" - "+j] = norm.interval(interval, loc=res, scale=np.std(data[i])/np.sqrt(len(data[j])));
    return dic

print(calcul_all_interval_confiance(data1, 0.95))