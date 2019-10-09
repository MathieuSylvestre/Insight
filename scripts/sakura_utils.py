import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import statistics
import xgboost

def get_all_windows(df, window_length, max_distance_to_peak, min_distance_to_peak = 0, col_to_drop=[]):
    """
    Gets all windows from a dataframe, where the rows of a dataframe are 
    properly ordered and should all be from the same city.
    
    Parameters
    ----------
    df: DataFrame
        a Pandas DataFrame with necessary columns, specifically
        'Date'
        'Is_Peak_Bloom'
        'Time_Since_Peak'
        'Time_To_Peak'
        'Latitude'
        'Day_Of_Year'
        as well as columns with the same name as entries of the list col_to_drop, if specified
    window_length: determines the number of days in considered windows
    max_distance_to_peak: determines windows to keep - windows that have a target of more than
        max_distance_to_peak are not returned
    min_distance_to_peak: analogous to max_distance_to_peak. 
        Default is min_distance_to_peak = 0
    col_to_drop: names of the columns in df that are dropped. 
        They are assumed refer to exist in df. Default is col_to_drop=[]
        
    Returns
    -------
    df_windows: a dataframe where the rows consist of all resulting windows
    """    
        
    #Create new dataframe for windows, shift and concatenate to get data from a sequence of days into a single row
    df_windows = df.drop(columns = ['Date','Is_Peak_Bloom','Time_Since_Peak','Time_To_Peak','Latitude','Day_Of_Year'],inplace=False)
    if(len(col_to_drop)>0):
        df_windows.drop(columns = col_to_drop,inplace = True)
    for i in range(1,window_length):
        df_temp = df.shift(-i)
        df_temp.drop(columns = ['Date','Is_Peak_Bloom','Time_Since_Peak','Time_To_Peak','Latitude','Day_Of_Year'],inplace=True)
        if(len(col_to_drop)>0):
            df_temp.drop(columns = col_to_drop,inplace = True)
        df_windows = pd.concat([df_windows,df_temp],axis=1)
    
    #include target and latitude on last frame
    df_windows = pd.concat([df_windows,df.shift(-window_length)],axis=1)
    
    #drop Is_Peak_Bloom
    #col_to_drop.append('Is_Peak_Bloom')
    if(len(col_to_drop)>0):
        df_windows.drop(columns = col_to_drop,inplace = True)
    
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
    
def normalize(x_train,x_test):
    """
    Normalizes training and test samples using sklearns fit_transform 
    and transform functions from the MinMaxScalar
    
    Parameters
    ----------
    x_train: training set to be normalized
    x_test: test set to be normalized
        
    Returns
    -------
    xn_train: normalized training set
    xn_test: normalized test set
    """  
        
    min_max_scaler = preprocessing.MinMaxScaler()
    xn_train = min_max_scaler.fit_transform(x_train)
    xn_test = min_max_scaler.transform(x_test)
    return xn_train, xn_test
    
#cross validate, separating train and test set by city. 
#df is a list of dataframes where each dataframe is associated to a city.
#mdl is the model to use
#col_to_drop is a list of columns to drop from the dataframe when creating the input set.
#k is the number of folds for cross validation
def cross_validate_by_city(df,mdl,target_col,col_to_drop = [],k=3):
    """
    Cross validation with separation of the cross-validation folds by city.
    Separation by city ensures that any set of overlapping windows are either 
    both in the training folds or both in the validation fold
    
    Parameters
    ----------
    df: dataframe containing the windowed features and corresponding target
    mdl: model for which cross-validation is performed. Assumed to have 'fit' 
        and 'predict', as in sklearn or xgboost
    target_col: name of column containing targets
    col_to_drop: list of names of columns to drop from df. 
        Default is col_to_drop=[]
    k: number of folds to use in cross validation. Default is k=3
    
    Returns
    -------
    maes: list containing the mean average error for each validation folds 
    """      
    
    #get number of cities in training data
    nb_train_cities = len(df)
    
    #use sklearn's module to facilitate cross validation
    cv = KFold(n_splits=k, shuffle=True)
    
    #store metrics to track performance
    maes = []
    
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

    return maes

def get_train_test_indices(nb_samples,test_size):
    """
    Obtain a random array of 0s and 1s with the fraction of 0s being approximately 
    equal to test_size. To be used to assign samples as training or test samples. 
    
    """
    train_indices = np.ones(nb_samples,dtype='int')
    
    #Set ~testsize of train indices to 0 to get a test set of ~testsize the data
    for i in range(int(test_size*nb_samples+1)):
        train_indices[i]=0
        
    #Permute to ensure randomness
    return np.random.permutation(train_indices)

def get_train_test_split_by_date_and_city(dfs,test_size=0.2):
    """
    Preprocessing to split data as training and testing data, ensuring
    that overlapping windows are either all used for training or for testing. 
    It is assumed that windows do not contain june 1st
    
    Parameters
    ----------
    dfs: list of dataframes containing windowed features and corresponding 
    target, where each dataframe in dfs were obtained from a different city
    test_size: fraction of the total number of windows that are assigned to the 
    test set.
    
    Returns
    -------
    train_dfs: list of dataframes to be used for the training set
    test_dfs: list of dataframes to be used for the test set
    """          
    train_dfs = []
    test_dfs = []
    df_years = []
    for df in dfs:
        #get earliest year for city
        year_init = int(df.Date.values[0].split(r'-',1)[0])
        for year in range(year_init,2018):
            upper_bound = str(year+1)+'-06-01'
            lower_bound = str(year)+'-06-01'
            df_year_low = df[(df['Date'] < upper_bound)]
            df_year = df_year_low[(df_year_low['Date'] > lower_bound)]
            if len(df_year) > 0:
                df_years.append(df_year)
                
    #assign city/year pairs to train or test set
    nb_years = len(df_years)
    train_indices = get_train_test_indices(nb_years,test_size)
    for i in range(nb_years):
        if train_indices[i] == 1:
            train_dfs.append(df_years[i])
        else:
            test_dfs.append(df_years[i])
    return train_dfs, test_dfs


def get_lists_of_windows(window_length, max_distance_to_peak, Cities, path, col_to_drop=[]):
    """
    Get all windows from all cities listed in Cities.
    
    Parameters
    ----------
    window_length: length of windows to be extracted from the data
    max_distance_to_peak: determines windows to keep - windows that have a 
        target of more than max_distance_to_peak are not returned
    Cities: list of names of cities for which the data should be extracted
    path: relative address of folder where the files are stored
    col_to_drop: names of the columns that are to be dropped. They are assumed 
        refer to exist. Default is col_to_drop=[]
    
    Returns
    -------
    dfs: list of dataframes, where each dataframe contains the windows of each 
        city in Cities
    """    
    
    dfs = []
    for city in Cities:
        #Load city related data and use the city as an index
        filepath = path + city + '_daily.csv'
        df = pd.read_csv(filepath)
        #store windows
        dfs.append(get_all_windows(df, window_length, max_distance_to_peak, 0, col_to_drop))
    return dfs

#main preprocessing tool once training and test set have defined
#   Assumed that train_df and test_df are lists of dataframes
def prepare_data_for_training(train_df, test_df, drop_Time_Since_Peak = False, drop_Day_Of_Year = True, drop_Latitude = False, return_numpy = True, weights = None):
    #Concatenate training and test samples
    train_all_df = pd.concat(train_df)
    test_all_df = pd.concat(test_df)
    
    #Set up for model and normalize as per common practice
    train_all_df = train_all_df.sample(frac=1).reset_index(drop=True)
    
    #Drop columns according to the arguments passed
    col_to_drop = ['Date','Is_Peak_Bloom','Time_To_Peak']
    if drop_Time_Since_Peak:
        col_to_drop.append('Time_Since_Peak')
    if drop_Day_Of_Year:
        col_to_drop.append('Day_Of_Year')    
    if  drop_Latitude:
        col_to_drop.append('Latitude')   
    y_train = train_all_df.Time_To_Peak.values
    train_dropped_columns = train_all_df.drop(columns = col_to_drop)
    x_train = train_dropped_columns.values
    
    #No need to shuffle test set. Not shuffling facilitates plotting predictions elegantly 
    y_test = test_all_df.Time_To_Peak.values    
    x_test = test_all_df.drop(columns = col_to_drop).values

    #Normalize input data
    xn_train, xn_test = normalize(x_train, x_test)
    
    if return_numpy:
        return xn_train, y_train, xn_test, y_test
    else:
        #make [xn_train,y_train] and [xn_test,y_test] dataframes
        n_train_df = pd.DataFrame(xn_train)
        n_test_df = pd.DataFrame(xn_test)
        n_train_df['Target'] = y_train.tolist()
        n_test_df['Target'] = y_test.tolist()   
        if weights != None:
            n_train_df['Weights'] = n_train_df.apply(lambda x: 1/(1+0.01*x.Target), axis=1)
            n_test_df['Weights'] = n_test_df.apply(lambda x: 1/(1+0.01*x.Target), axis=1)
    return n_train_df, n_test_df

#Plots days til bloom. y_true should be a decreasing sequence, y_pred
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

    plt.scatter(y_true[-last_day - days_plotted:-last_day],
                y_pred[-last_day - days_plotted:-last_day])
    plt.xlabel('True number of days to peak bloom')
    plt.ylabel('Predicted number of days to bloom')
    plt.show()
    
#returns the mean average error between y_pred and y_true as a vector, where
#index refers to the number of days til peak bloom
def plot_mae_vs_y_true(y_pred,y_true,points_to_plot = None):
    error = np.abs(y_pred-y_true)
    error_sum = np.zeros(int(np.max(y_true)+1))
    counts = np.ones(int(np.max(y_true)+1))
    for i in range(len(y_true)):
        if not (np.isinf(error[int(y_true[i])]) or np.isnan(error[int(y_true[i])])):
            error_sum[int(y_true[i])] += error[i]
            counts[int(y_true[i])] += 1
    avg_error = error_sum/counts
    
    plt.figure()
    if points_to_plot == None:
        plt.plot(avg_error)
    else:
        plt.plot(avg_error[:points_to_plot])
    plt.xlabel('Days til peak bloom')
    plt.ylabel('Average Error (Days)')
    
    return avg_error

#returns the median absolute difference between y_pred and y_true as a vector, 
#where the index of med_error refers to the number of days til peak bloom
#if name isn't null, the plot is saved as an eps file 
def plot_median_error_vs_y_true(y_pred,y_true,points_to_plot = None, name = None):
    error = np.abs(y_pred-y_true)
    error_lists = [ [] for i in range(int(np.max(y_true)+1)) ]
    for i in range(len(y_true)):
        if not (np.isinf(error[int(y_true[i])]) or np.isnan(error[int(y_true[i])])):
            error_lists[int(y_true[i])].append(error[i])
    
    #get median for each list
    med_error = np.zeros(int(np.max(y_true)+1))
    for i in range(int(np.max(y_true))):
        med_error[i] = statistics.median(error_lists[i])
    
    #Plot results
    plt.figure()
    if points_to_plot == None:
        plt.plot(med_error)
    else:
        plt.plot(med_error[:points_to_plot])
    plt.xlabel('Days til peak bloom')
    plt.ylabel('Median Error (Days)')
    plt.gca().invert_xaxis()
    plt.gca().set_ylim(bottom=-0.1)
    if name != None:
        plt.savefig(name, format='eps', dpi=1000)
    return med_error
    
def Cross_Validate(df, window_length, max_distance_to_peak):
    #Do k-fold cross validation and show metrics - comment out when not doing hyperparameter tuning
    max_depths = []
    avg_maes = []
    hyperparams = []    

    for depth in range(5,9):
        
        for n_estimators_index in range(2,6):
            
            n_estimators = 50 * n_estimators_index
            str_hyperparams = 'dist_to_peak: ' + str(max_distance_to_peak) + '\nwindow_length: ' + str(window_length) + '\nmax_depth: ' + str(depth) + '\nn_estimators: ' + str(n_estimators)
            print(str_hyperparams)
            
            #Define model
            xgbr = xgboost.XGBRegressor(max_depth=depth,
                                        n_estimators = n_estimators,
                                        objective = 'reg:squarederror')
            
            #Return metrics related to cross validation by city
            maes, r2s = cross_validate_by_city(df = df,
                                               mdl = xgbr,
                                               target_col='Time_To_Peak',
                                               col_to_drop=['Date'])
            
            print('\nMAEs and R^2: max_depth = ' + str(depth))
            print('Average MAE: ' + str(np.mean(maes)) + 'all: ' + str(maes))
            
            max_depths.append(depth)
            avg_maes.append(np.mean(maes))
            hyperparams.append(str_hyperparams)
    
    #Plot evolution of test set error as a function of the model complexity
    plt.plot(max_depths,avg_maes)
    plt.xlabel('Model complexity')
    plt.ylabel(r'Test fold MAE')
    plt.show()
    
    return avg_maes, hyperparams