#sakura utils

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

import matplotlib.pyplot as plt

#input all data from a city. returns all time windows of length window_length
#time windows farther than max_distance_to_peak from the next peak bloom are discarded
#time windows closer than min_distance_to_peak are also discarded
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
    #Only keep windows with target less than max_distance_to_peak days away from target
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

#get a subset of a dataframe of windows based on the target value.
#samples with target values are dropped.
def drop_old_windows(df_windows,max_distance_to_peak):
    i_to_drop = df_windows[df_windows['Time_To_Peak'] > max_distance_to_peak].index
    df_windows.drop(i_to_drop , inplace=True)
    return df_windows
    
#Standar min_max normalozation, applying to the train set, use on the test set
def normalize(x_train,x_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    xn_train = min_max_scaler.fit_transform(x_train)
    xn_test = min_max_scaler.transform(x_test)
    return xn_train, xn_test
    
#cross validate, separating train and test set by city. 
#df is a list of dataframes where each dataframe is associated to a city.
#mdl is the model to use
#col_to_drop is a list of columns to drop from the dataframe when creating the input set.
#k is the number of folds for cross validation
def cross_validate_by_city(df,mdl,target_col,col_to_drop,k=3):
    
    #get number of cities in training data
    nb_train_cities = len(df)
    
    #use sklearn's module to facilitate cross validation
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
        
        #Set up for model and normalize
        col_to_drop.append(target_col) #in addition to predicitive features, 
                                       #remove the target column from input features 
        y_cv_test = test_cv_df[target_col].values
        x_cv_test = test_cv_df.drop(columns = col_to_drop).values
        
        y_cv_train = train_cv_df.Time_To_Peak.values
        x_cv_train = train_cv_df.drop(columns = ['Date','Time_To_Peak']).values
        
        #normalize
        xn_cv_train, xn_cv_test = normalize(x_cv_train,x_cv_test)
        
        #fit and get metrics on left-out fold
        mdl.fit(xn_cv_train,y_cv_train)
        y_pred = mdl.predict(xn_cv_test)
        maes.append(median_absolute_error(y_pred, y_cv_test))
        r2s.append(r2_score(y_pred, y_cv_test))

    return maes, r2s

#separate cities as training cities and test cities
def get_train_test_city_indices(nb_cities,test_size):    
    #Do k-fold cross validation for model selection
    train_indices = np.ones(nb_cities,dtype='int')
    
    #Set ~20% of train indices to 0 to get a test set of ~20% the data
    for i in range(int(test_size*nb_cities+1)):
        train_indices[i]=0
        
    #Permute to ensure randomness
    return np.random.permutation(train_indices)

#split train and test set. traindices is a list of length number of cities.
#train_indices[i] should be 1 if the city is assigned to the train set and 0
#otherwise
def get_train_test_split_by_city(train_indices,dfs):
    train_dfs = []
    test_dfs = []
    for i in range(len(train_indices)):
        if train_indices[i] == 1:
            train_dfs.append(dfs[i])
        else:
            test_dfs.append(dfs[i])
        
    return train_dfs, test_dfs

#get all windows from all cities listed in Cities
#returned windows are of length window_length
#path is the folder in which the cleaned data is contained
def get_lists_of_windows(window_length, max_distance_to_peak, Cities, path):
    dfs = []
    for city in Cities:
        #Load city related data and use the city as an index
        filepath = path + city + '_daily.csv'
        df = pd.read_csv(filepath)
        #store windows
        dfs.append(get_all_windows(df,window_length,max_distance_to_peak))
    return dfs

def prepare_data_for_training(train_df, test_df):
    #Concatenate training and test samples
    train_all_df = pd.concat(train_df)
    test_all_df = pd.concat(test_df)
    
    #Set up for model and normalize as per common practice
    train_all_df = train_all_df.sample(frac=1).reset_index(drop=True)
    
    y_train = train_all_df.Time_To_Peak.values
    x_train = train_all_df.drop(columns = ['Date','Time_To_Peak']).values
    
    #No need to shuffle test set. Not shuffling facilitates plotting predictions
    #elegantly 
    y_test = test_all_df.Time_To_Peak.values
    x_test = test_all_df.drop(columns = ['Date','Time_To_Peak']).values
    
    #Normalize input data
    xn_train, xn_test = normalize(x_train, x_test)
    
    return xn_train, y_train, xn_test, y_test

#plots days til bloom. y_true should be a decreasing sequence, y_pred
#associated predictions
def plot_predictions_over_time(y_true,y_pred):
    y_true = np.flip(y_true)
    plt.scatter(y_true,y_pred)
    plt.xlabel('time (days)')
    plt.ylabel('Predicted days to bloom')
    plt.show()
    
#Might be removed
def plot_predictions_versus_true(y_true,y_pred, days_plotted = 60, years_back = 0):
    year_counter = 0
    last_day = 0
    first_day = 1
    while year_counter <= years_back:
        first_day += 1
        if first_day == len(y_true)-1:
            break
        elif y_true[-first_day] > y_true[-first_day-1]:
            year_counter +=1
            if year_counter <= years_back:
                last_day = first_day

#    plt.scatter(y_true[-first_day:-last_day],y_pred[-first_day:-last_day])
    plt.scatter(y_true[-last_day - days_plotted:-last_day],
                y_pred[-last_day - days_plotted:-last_day])
    plt.xlabel('True number of days to peak bloom')
    plt.ylabel('Predicted number of days to bloom')
    plt.show()