import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import csv  
from pandas import read_csv
from datetime import datetime, timedelta
import numpy as np
# load dataset
series = read_csv('confirmed.csv', header=0, parse_dates=True, squeeze=True)
print(len(series['date']))
X = (series['total'].values)
train, test = X[1:len(X)-12], X[len(X)-12:]
# train autoregression
window = 5
model = AutoReg(train, lags=5)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true)+1))) * 100
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# plot
base= datetime.strptime((series['date'][len(series)-12]), '%Y-%m-%d')
delta = timedelta(days=1)
target_date =[(base + timedelta(days=i)).strftime("%d %b %y")  for i in range(12)]

plt.plot(test)
plt.plot(predictions, color='red')
plt.ylabel("Number of cases"); plt.xlabel("Date")
plt.xticks(list(range(0,12)),target_date, rotation=32, ha='right')
plt.savefig('case.jpg')