#set up data as a window from start time end-time
import pandas as pd
import numpy as np
from sklearn import preprocessing
#from sklearn.model_selection import cross_val_score
import xgboost
#import model
#from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import sakura_utils as sutils

import matplotlib.pyplot as plt

import time 
import pickle

#Cities considered (more to come)
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha','washington']

#Do k-fold cross validation for model selection
nb_cities = len(Cities)

# assign cities to train and test cities
train_indices = sutils.get_train_test_city_indices(nb_cities,test_size = 0.2)
train_df, test_df = sutils.get_train_test_split_by_city(train_indices, dfs_w)

window_length = 50
max_distance_to_peak = 150

dfs_w = sutils.get_lists_of_windows(window_length, 
                                    max_distance_to_peak, 
                                    Cities, 
                                    path = '../data/cleaned/')
train_df, test_df = sutils.get_train_test_split_by_city(train_indices, dfs_w)

##Do cross validation and show metrics - comment out when not doing hyperparameter tuning
#max_depths = []
#avg_maes = []
#avg_r2 = []
#hyperparams = []
#
#for dist_index in range(2):
#    if dist_index == 0:
#        max_distance_to_peak = 40
#    else:
#        max_distance_to_peak = 100
#        
#    for window_length_index in range(2,5):
#        window_length = 25 * window_length_index
#    #    window_length = 100
#    #    max_distance_to_peak = 40
#        dfs_w = sutils.get_lists_of_windows(window_length, 
#                                            max_distance_to_peak, 
#                                            Cities, 
#                                            path = '../data/cleaned/')
#        for depth in range(3,10):
#            
#            for n_estimators_index in range(2,7):
#                
#                n_estimators = 50 * n_estimators_index
#                str_hyperparams = 'dist_to_peak: ' + str(max_distance_to_peak) + '\nwindow_length: ' + str(window_length) + '\nmax_depth: ' + str(depth) + '\nn_estimators: ' + str(n_estimators)
#                print(str_hyperparams)
#                
#                #Define model
#                xgbr = xgboost.XGBRegressor(max_depth=depth,
#                                            n_estimators = n_estimators,
#                                            objective = 'reg:squarederror')
#                
#                #Return metrics related to cross validation by city
#                maes, r2s = sutils.cross_validate_by_city(df = train_df,
#                                                          #mdl = regr, 
#                                                          mdl = xgbr,
#                                                          target_col='Time_To_Peak',
#                                                          col_to_drop=['Date'])
#                
#                print('\nMAEs and R^2: max_depth = ' + str(depth))
#                print('Average MAE: ' + str(np.mean(maes)) + 'all: ' + str(maes))
#                print('Average R^2: ' + str(np.mean(r2s)) + 'all: ' + str(r2s))
#                
#                max_depths.append(depth)
#                avg_maes.append(np.mean(maes))
#                avg_r2.append(np.mean(r2s))
#                hyperparams.append(str_hyperparams)
#
##Plot evolution of test set error as a function of the model complexity
#plt.plot(max_depths,avg_maes)
##plt.plot(max_depths,avg_r2)
#plt.xlabel('Model complexity')
#plt.ylabel(r'Test set MAE and $R^2$')
#plt.show()

#get data for model training
xn_train, y_train, xn_test, y_test = sutils.prepare_data_for_training(train_df,test_df)

#Didn't seem to work well
sample_weight = np.ones(len(y_train))

#Train model with all training data
print('With XGBRegressor')
xgbr = xgboost.XGBRegressor(max_depth=8,
                            n_estimators = 100,
                            objective = 'reg:squarederror')
xgbr.fit(xn_train,y_train,sample_weight)

filename = 'model_short' + str(int(time.time())) + '.sav'
pickle.dump(xgbr, open(filename, 'wb'))

#make predicitions
print('Test set predictions')
y_pred = xgbr.predict(xn_test)
print('MAE : ' + str(median_absolute_error(y_test,y_pred)))
print('R^2 : ' + str(r2_score(y_test,y_pred)))

#Plot predictions over time for the last year of a city from the test set
points_to_plot = 40#+2*152
dist_from_end = 0#+2*152
if dist_from_end < 1:
    sutils.plot_predictions_over_time(y_test[-points_to_plot:],
                                      y_pred[-points_to_plot:])
else:
    sutils.plot_predictions_over_time(y_test[-points_to_plot:-dist_from_end],
                                      y_pred[-points_to_plot:-dist_from_end])

#Plot true vs predicted for some year
sutils.plot_predictions_versus_true(y_test,y_pred,100,2)

print('Train set predictions')
y_pred_train = xgbr.predict(xn_train)
print('MAE : ' + str(median_absolute_error(y_train,y_pred_train)))
print('R^2 : ' + str(r2_score(y_train,y_pred_train)))

#TODO: get two models working. Once this is done, this code will be moved elsewhere
if False:
    xn_train_long, y_train_long, xn_test_long, y_test_long = sutils.prepare_data_for_training(train_df_long,test_df_long)
    xn_train_short, y_train_short, xn_test_short, y_test_short = sutils.prepare_data_for_training(train_df_short,test_df_short)
    
    loaded_model_short = pickle.load(open('model_short.sav', 'rb'))
    loaded_model_long = pickle.load(open('model_long.sav', 'rb'))
    
    y_pred_long = loaded_model_long.predict(xn_test_long)
    y_pred_short = loaded_model_short.predict(xn_test_short)
    
    #output from most appropriate model
    y_pred = y_pred_long
    for i in range(len(y_pred)):
        if y_pred_long > 25:
            y_pred = y_pred_short
    