#sakura utils

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import xgboost
import model
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

import matplotlib.pyplot as plt

#Currently not used
#start_date and end_date must be a string of the form yyyy-mm-dd
def get_window_from_dates(df, start_date, end_date, columns_to_drop, keep_latitude = True):

    #Find start date and end date index
    i_start = df[df.Date == start_date].index
    i_end = df[df.Date == end_date].index
    
    df_window = df[i_start[0]:i_end[0]+1]
    df_window.drop(columns = columns_to_drop, inplace=True)
    
    print('dropped')
    window = df_window.to_numpy().flatten()

    print('flattened')    
    if keep_latitude:
        latitude = df.Latitude[-1]
        window = np.concatenate((latitude,window))
    return window

#input all data from a city
def get_all_windows(df,window_length,max_distance_to_peak,min_distance_to_peak=0):
    
    #Create new dataframe for windows, shift and concatenate to get data from a sequence of days into a single row
    df_windows = df.drop(columns = ['Date','Is_Peak_Bloom','Time_Since_Peak','Time_To_Peak','Latitude'],inplace=False)
    for i in range(1,window_length):
        df_temp = df.shift(-i)
        df_temp.drop(columns = ['Date','Is_Peak_Bloom','Time_Since_Peak','Time_To_Peak','Latitude'],inplace=True)
        df_windows = pd.concat([df_windows,df_temp],axis=1)
    
    #include target and latitude on last frame
    df_windows = pd.concat([df_windows,df.shift(-window_length)],axis=1)
    
    df_windows = df_windows.dropna() #NaNs were generate as a result of the shifting
    
    
    #Delete these row indexes from dataFrame    
    #Only keep windows with target less than 150 days away
    i_to_drop = df_windows[df_windows['Time_To_Peak'] > max_distance_to_peak].index
    df_windows.drop(i_to_drop , inplace=True)
    if min_distance_to_peak > 0:
        i_to_drop = df_windows[df_windows['Time_To_Peak'] >= min_distance_to_peak].index
        df_windows.drop(i_to_drop , inplace=True)
    i_to_drop = df_windows[df_windows['Time_To_Peak'] < 0 ].index #Sakura hasn't occured yet
    df_windows.drop(i_to_drop , inplace=True)     
    i_to_drop = df_windows[df_windows['Time_Since_Peak'] < 0 ].index #Latest date of Sakura unknown
    df_windows.drop(i_to_drop , inplace=True)
    return df_windows

#Target is 
def cross_validate_by_city(df,mdl,target_col,col_to_drop,k=3):
    
    #get number of cities in training data
    nb_train_cities = len(df)
    
    #define how cross validation will work
    cv = KFold(n_splits=k, shuffle=True)
    
    #store metrics to track performance
    maes = []
    r2s = []
    
    #separate k_folds, with test and train indices. Train model for each and
    #return score for the left-out fold
    for train_cities, test_cities in cv.split(np.arange(nb_train_cities)): 
        
        #create training and test sets for the cross validation
        train_cv_df = []
        test_cv_df = []
        for index in train_cities:
            train_cv_df.append(df[index])
        for index in test_cities:
            test_cv_df.append(df[index])
            
        #concatenate into a training set and test sets and shuffle for training
        train_cv_df = pd.concat(train_cv_df)
        train_cv_df = train_cv_df.sample(frac=1).reset_index(drop=True)
        test_cv_df = pd.concat(test_cv_df)
        test_cv_df = test_cv_df.sample(frac=1).reset_index(drop=True)
        
        #Set up for model and normalize as per common practice
        
        col_to_drop.append(target_col)
        y_cv_test = test_cv_df[target_col].values
        x_cv_test = test_cv_df.drop(columns = col_to_drop).values
 
#        y_cv_test = test_cv_df.Time_To_Peak.values
#        x_cv_test = test_cv_df.drop(columns = ['Date','Time_To_Peak']).values
       
        y_cv_train = train_cv_df.Time_To_Peak.values
        x_cv_train = train_cv_df.drop(columns = ['Date','Time_To_Peak']).values
        
        #normalize
        min_max_scaler = preprocessing.MinMaxScaler()
        xn_cv_train = min_max_scaler.fit_transform(x_cv_train)
        xn_cv_test = min_max_scaler.transform(x_cv_test)
        
        #fit and get metrics on left-out fold
        mdl.fit(xn_cv_train,y_cv_train)
        y_pred = mdl.predict(xn_cv_test)
        maes.append(median_absolute_error(y_pred, y_cv_test))
        r2s.append(r2_score(y_pred, y_cv_test))

    return maes, r2s

#plots days til bloom. y_true should be a decreasing sequence, y_pred
#associated predictions
def plot_predictions_over_time(y_true,y_pred):
    y_true = np.flip(y_true)
    plt.scatter(y_true,y_pred)
    plt.xlabel('time (days)')
    plt.ylabel('Predicted days to bloom')
    plt.show()
    