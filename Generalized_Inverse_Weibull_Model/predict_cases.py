import pandas as pd
from pandas import read_csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
from numpy import inf
from math import exp
from datetime import timedelta
import matplotlib.patheffects as PathEffects
from scipy.special import softmax
import warnings
import os
from tqdm import tqdm
import math
from scipy.stats import pearsonr, spearmanr

warnings.simplefilter("ignore")


plt.rcParams["text.usetex"] = True
series = read_csv('confirmed.csv', header=0, parse_dates=True, squeeze=True)

def weib(x, k, a, b, g):
	return k * g * b * (a ** b) * np.exp(-1 * g * ((a / x)  ** b)) / (x ** (b + 1))


def seriesIterativeCurveFit(func, xIn, yIn, start):
	res = []
	for ignore in tqdm(list(range(10, 0, -1)), ncols=80):
		x = xIn[:-1*ignore]; y = yIn[:-1*ignore]
		outliersweight = None
		for i in range(10):
			popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, absolute_sigma=True, maxfev=1000000)
			pred = np.array([func(px, *popt) for px in x])
			old = outliersweight
			outliersweight = np.abs(pred - y)
			outliersweight = outliersweight - np.tanh(outliersweight)
			outliersweight = outliersweight / np.max(outliersweight)
			outliersweight = softmax(1 - outliersweight)
			if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
		pred = [func(px, *popt) for px in xIn]
		res.append((mean_absolute_percentage_error(yIn, pred), popt, pcov, ignore))
	errors = [i[0] for i in res]
	val = res[errors.index(min(errors))]
	return val[1], val[2]

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true)+1))) * 100

ignore = -1
training_data = -12
for country in ['India']:
    try:
        data = (series['total'].values).tolist()
        start= datetime.strptime((series['date'][0]), '%Y-%m-%d')
        days = len(series['date'])
        plt.figure(figsize=(7,4))
        x = list(range(len(data)))
        xlim = len(data)*2
        datacopy = np.absolute(np.array(deepcopy(data[1:training_data])))
        popt, pcov = seriesIterativeCurveFit(weib, x[1:training_data], datacopy, [160000, 14, 4, 500])
        y = [weib(px, *popt) for px in x[1:]]
        mapeCase = mean_absolute_percentage_error(data[1:], y)
        mape_error_case = mean_absolute_percentage_error(data[training_data:], y[training_data:])
        pred = [weib(px, *popt) for px in list(range(xlim))[1:]]
        plt.plot(list(range(xlim))[1:], pred, color='black', label='Robust Weibull Prediction (case)')
        _ = plt.bar(x[:training_data], data[:training_data], width=1, color='blue',edgecolor='black', linewidth=0.01, alpha=0.2, label='Train Data (case)')
        _ = plt.bar(x[training_data:], data[training_data:], width=1, color='red',edgecolor='black', linewidth=0.01, alpha=0.2,label='Test Data (case)')
        plt.ylabel("Number of cases"); plt.xlabel("Date"); plt.tight_layout()
        plt.legend(loc='best');	plt.title(country)
        plt.xticks(list(range(0,xlim,30)), [(start+timedelta(days=i)).strftime("%d %b %y") for i in range(0,xlim,30)], rotation=45, ha='right')
        plt.savefig('cases.png')
        print("MAPE",mapeCase)
        print("----", country)
    except Exception as e:
        print(str(e))
        raise(e)
        pass
