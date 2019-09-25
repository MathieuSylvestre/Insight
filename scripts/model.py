#for training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.datasets import make_regression

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

import matplotlib.pyplot as plt

class WindowedRegressionModel():

    def __init__(self,regr = 'Random Forest Regressor', max_depth=4, random_state=0, n_estimators=250):     
        if regr == 'Linear Regressor':
            self.regr = LinearRegression()
        else:
            self.regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)

        
    def train(self,x_train, y_train):
        self.regr.fit(x_train, y_train) 
        
    def predict(self,x):
        return self.regr.predict(x)
        
    def get_mae(self,y,y_pred):
        return median_absolute_error(y,y_pred)
    
    def get_r2(self,y,y_pred):
        return r2_score(y,y_pred)

class LSTMModel():

    def __init__(self, input_shape, num_LSTM_units=32, num_dense_layers=3, is_regression =True, activation = 'binary_crossentropy'):
        
        self.model = Sequential()
        #self.model.add(LSTM(num_units, input_shape=(12,23)))
        self.model.add(LSTM(num_LSTM_units, input_shape=input_shape))
        for i in range(num_dense_layers):
            self.model.add(Dense(32, activation='relu'))
        if is_regression:
            self.model.add(Dense(1))
        else:
            self.model.add(Dense(1), activation=activation)

    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=50, batch_size=1, loss='mae'):
        self.model.compile(loss=loss, optimizer='adam', metrics=['mean_absolute_error'])
        if x_test is None:
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, )
            self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, callbacks=[es])
        
    def predict(self,x):
        return self.model.predict(x) 
    
    #For regression
    def get_mae(self,x,y,y_pred = None):
        if y_pred is None:
            self.predict(self,x)
        return median_absolute_error(y_pred,y)
    
    def get_mae_thresholded(self,x,y,threshold): #Probably not useful
        y_pred = self.model.predict(x).T[0]
        sum_dif = 0
        count= 0
        for i in range(len(y)):
            if max(y[i],y_pred[i]) > threshold:
                sum_dif+= abs(y[i]-y_pred[i])
                count+=1
        return sum_dif/count