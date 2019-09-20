#for training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.datasets import make_regression

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

import matplotlib.pyplot as plt

#X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regrRF = RandomForestRegressor(max_depth=4, random_state=0,
                             n_estimators=250)

print('Linear Regression')
regrLR = LinearRegression()
regrLR.fit(x_train_normalized, y_train) 

y_train_pred = regrLR.predict(x_train_normalized)

print('Train result')
plt.plot(y_train,y_train_pred,'o')
print('R2_score: ' + str(r2_score(y_train,y_train_pred)))
print('MAE     : ' + str(median_absolute_error(y_train,y_train_pred)))

plt.figure()
y_test_pred = regrLR.predict(x_test_normalized)
print('Test result')
plt.plot(y_test,y_test_pred,'o')
print('R2_score: ' + str(r2_score(y_test,y_test_pred)))
print('MAE     : ' + str(median_absolute_error(y_test,y_test_pred)))


print('Random Forest')
regrRF.fit(x_train_normalized, y_train) 

y_train_pred = regrRF.predict(x_train_normalized)
print('Train result')
plt.plot(y_train,y_train_pred,'o')
print('R2_score: ' + str(r2_score(y_train,y_train_pred)))
print('MAE     : ' + str(median_absolute_error(y_train,y_train_pred)))
plt.figure()
y_test_pred = regrRF.predict(x_test_normalized)
print('Test result')
plt.plot(y_test,y_test_pred,'o')
print('R2_score: ' + str(r2_score(y_test,y_test_pred)))
print('MAE     : ' + str(median_absolute_error(y_test,y_test_pred)))