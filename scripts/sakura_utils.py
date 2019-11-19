import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import statistics
import xgboost
import joblib

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
        df_windows.drop(columns = col_to_drop, inplace = True)
    
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
    
def normalize(x_train,x_test,save_transform = False):
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
    if save_transform:
        joblib.dump(min_max_scaler, "min_max_scaler")
    return xn_train, xn_test

def cross_validate_from_list(dfs, mdl, target_col, weighting = None, col_to_drop = [], k=5):
    """
    Cross validation with separation of the cross-validation folds based on dfs.
    dfs is a list of dataframes, and the samples from each dataframe is grouped 
    together in the same fold. Used to ensure that all overlapping windows are 
    either all in the training folds or all in the validation fold.
    
    Parameters
    ----------
    dfs: list of dataframes containing the windowed features and corresponding target
    mdl: model for which cross-validation is performed. Assumed to have 'fit' 
        and 'predict', as in sklearn or xgboost
    target_col: name of column containing targets
    weighting: function that takes in an array of target values and return a list
        of weights corresponding to each sample
    col_to_drop: list of names of columns to drop from df. 
        Default is col_to_drop=[]
    k: number of folds to use in cross validation. Default is k=5
    
    Returns
    -------
    maes: list containing the mean average error for each validation folds 
    """      
    
    #get number of cities in training data
    nb_groups = len(dfs)
    
    #use sklearn's module to facilitate cross validation
    cv = KFold(n_splits=k, shuffle=True)
    
    #store metrics to track performance
    maes = []
    
    #separate k_folds, with test and train indices. Train model for each and
    #return score for the left-out fold
    for train_folds, test_folds in cv.split(np.arange(nb_groups)): 
        
        #create training and test sets for the cross validation
        train_cv_df = []
        test_cv_df = []
        for index in train_folds:
            train_cv_df.append(dfs[index])
        for index in test_folds:
            test_cv_df.append(dfs[index])
            
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
        x_cv_train = train_cv_df.drop(columns = col_to_drop).values#['Date','Time_To_Peak']).values
        
        #normalize
        xn_cv_train, xn_cv_test = normalize(x_cv_train,x_cv_test)
        
        if weighting is None:
            #fit and get metrics on left-out fold
            mdl.fit(xn_cv_train, y_cv_train)
        else:
            #compute weighting for the training set then fit and get metrics on left-out fold
            weights = weighting(y_cv_train)
            mdl.fit(xn_cv_train, y_cv_train, weights)
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

def get_sample_weights(days_before_offset,x,scale_factor=100):
    """
    Get the sample weight for a sample, based on actual time to peak bloom and
    an inputed offset. For days after offset, downweight by factor 1/target
    
    Parameters
    ----------
    days_before_offset: time before peak bloom for which no downweighting is 
        applied
    x: true value (target, days to peak bloom)
    scale_factor: scaling factor on weights, default 100
    
    Returns
    -------
    weight sample for the given sample 
    """       
    days_before_offset = max(1, days_before_offset) 
    if x < days_before_offset:
        return scale_factor
    else:
        return scale_factor * days_before_offset/x

#main preprocessing tool once training and test set have defined
#   Assumed that train_df and test_df are lists of dataframes
def prepare_data_for_training(train_df, test_df, drop_Time_Since_Peak = False, drop_Day_Of_Year = True, drop_Latitude = False, return_numpy = True, weight_delays = None, save_transform=False):
    """
    From training and test sets stored as lists of dataframes, performs normalization
    and returns training and test sets ready for training
    
    Parameters
    ----------
    train_df: list of dataframes containing windows to be used for training
    test_df: list of dataframes containing windows to be used for testing
    drop_Time_Since_Peak: boolean, whether the column Time_Since_Peak should be 
        dropped from the feature set
    drop_Day_Of_Year: boolean, whether the column Day_Of_Year should be 
        dropped from the feature set    
    drop_Latitude: boolean, whether the column Latitude should be 
        dropped from the feature set   
    return_numpy: boolean, whether function should return x_train/test and 
        y_train/test as numpy arrays or pandas dataframes
    weight_delays: if using delays, set how much delay (time from peak) until
        downweighting samples. If not None, adds column 'Weights' containing 
        how much each sample should be weighted. Only applies if return_numpy
        is False. Default is None
    
    Returns
    -------
    if return_numpy == True
        xn_train: numpy array, training set, normalized
        y_train: numpy array, training set targets
        xn_test: numpy array, test set, normalized
        y_test: numpy array, test set targets
    else
        n_train: dataframe containing all training samples - with features and samples
        n_test: dataframe containing all testing samples - with features and samples
    """    
    
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
    xn_train, xn_test = normalize(x_train, x_test,save_transform=True)
    
    if return_numpy:
        return xn_train, y_train, xn_test, y_test
    else:
        #make [xn_train,y_train] and [xn_test,y_test] dataframes
        n_train_df = pd.DataFrame(xn_train)
        n_test_df = pd.DataFrame(xn_test)
        n_train_df['Target'] = y_train.tolist()
        n_test_df['Target'] = y_test.tolist()   
        #add weights
        if weight_delays != None:
            n_train_df['Weights'] = n_train_df.apply(lambda x: get_sample_weights(weight_delays,x.Target), axis=1)
            n_test_df['Weights'] = n_test_df.apply(lambda x: get_sample_weights(weight_delays,x.Target), axis=1)            
    return n_train_df, n_test_df

def plot_predictions_over_time(y_true,y_pred):
    """
    Plots predictions as a function of the true value
    
    Parameters
    ----------
    y_pred: predicted values, array
    y_true: actual values, must be array of same length as y_pred
    
    """
    y_true = np.flip(y_true)
    plt.scatter(y_true,y_pred)
    plt.xlabel('time (days)')
    plt.ylabel('Predicted days to bloom')
    plt.show()
    
    
def plot_predictions_versus_true(y_true,y_pred, days_plotted = 60, years_back = 0):
    """
    Plot predictions as a function of the true value for a set number of 
    consecutive days ending at 0 days to peak bloom.
    
    Parameters
    ----------
    y_pred: predicted values, array
    y_true: actual values, must be array of same length as y_pred. Assumed to be
        in order, i.e. for a given city/year, dates are all together in an 
        ascending order
    days_plotted: number of days plotted
    years_back: number of city/years to go back and plot
    
    """
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

def plot_mae_vs_y_true(y_pred,y_true,points_to_plot = None):
    """
    Computes the mean absolute error between y_pred and y_true as a vector, 
    where the index of med_error refers to the number of days til peak bloom.
    Also produces a plot, and if name isn't null, the plot is saved as an eps 
    file.
    
    Parameters
    ----------
    y_pred: predicted values, array
    y_true: actual values, must be array of same length as y_pred
    points_to_plot: number of days (points) to plot. default is max value in y_true
    name: name of file (if saved). Defaut None (not saved)
    
    Returns
    -------
    avg_error: array of median error indexed by the number of days to peak bloom
    
    """
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

def plot_median_error_vs_y_true(y_pred,y_true,points_to_plot = None, name = None):
    """
    Computes the median absolute error between y_pred and y_true as a vector, 
    where the index of med_error refers to the number of days til peak bloom
    if name isn't null, the plot is saved as an eps file
    
    Parameters
    ----------
    y_pred: predicted values, array
    y_true: actual values, must be array of same length as y_pred
    points_to_plot: number of days (points) to plot. default is max value in y_true
    name: name of file (if saved). Defaut None (not saved)
    
    Returns
    -------
    med_error: array of median error indexed by the number of days to peak bloom
    
    """
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
    plt.ylabel('Median Absolute Error (Days)')
    plt.gca().invert_xaxis()
    plt.gca().set_ylim(bottom=-0.1)
    if name != None:
        plt.savefig(name, format='eps', dpi=1000)
    return med_error
    
def grid_search_cross_validation_XGBR(dfs, depths, estimators, weighting = None, target_col='Time_To_Peak', col_to_drop = [], k = 5, verbose= False):
    """
    Cross validation of xgboostRegressor with separation of the cross-validation folds based on dfs.
    dfs is a list of dataframes, and all samples of a dataframe in dfs are in the same CV fold.
    This is used to get separation by city and year, since separation by city/year ensures that any 
    set of overlapping windows are either both in the training folds or both in the validation fold. 
    Grid search is performed over the values in depths and estimators.
    
    Parameters
    ----------
    dfs: list of dataframes containing the windowed features and corresponding
        target. Each dataframe in the list is from a same unique year/city pair
    depths: list of integers containing the depths to be used in cross validating 
        xgboost regressor
    estimators: list of integers containing the number of estimators to be used 
        in cross validating xgboost regressor
    weighting: function that takes in an array of target values and return a list
        of weights corresponding to each sample. Default is None
    target_col: name of column containing targets. Default is 'Time_To_Peak'
    col_to_drop: list of names of columns to drop from df. 
        Default is col_to_drop=[]
    k: number of folds to use in cross validation. Default is k=5
    verbose: boolean. If True, prints progress and performance. Default is False
    
    Returns
    -------
    maes: list containing the mean average error for each validation folds 
    hyperparams: string describing the hyperparameters for each tested set of hyperparameter values
    """     
    
    #store hyperparaters used to track performance
    max_depths = []
    avg_maes = []
    hyperparams = []    

    for depth in depths:
        
        for n_estimators in estimators:
            
            str_hyperparams = 'max_depth: ' + str(depth) + '. n_estimators: ' + str(n_estimators)
            if verbose: print(str_hyperparams) #to show progress
            
            #Define model
            xgbr = xgboost.XGBRegressor(max_depth=depth,
                                        n_estimators = n_estimators,
                                        objective = 'reg:squarederror')
            
            #Perform cross validation for the defined model and get its performance metrics
            maes = cross_validate_from_list(dfs = dfs, 
                                            mdl = xgbr, 
                                            target_col=target_col, 
                                            weighting = weighting,
                                            col_to_drop=col_to_drop,
                                            k = k)
            if verbose:
                #print('\nMAEs and R^2: max_depth = ' + str(depth))
                print(str_hyperparams + '. Average MAE: ' + str(np.mean(maes)) + '. All: ' + str(maes))
            
            max_depths.append(depth)
            avg_maes.append(np.mean(maes))
            hyperparams.append(str_hyperparams)
    
    #Plot evolution of test set error as a function of the model complexity
    plt.plot(max_depths,avg_maes)
    plt.xlabel('Model complexity') #actual model complexity only if depths or estimators is of length 1
    plt.ylabel(r'Test fold MAE')
    plt.show()
    
    return avg_maes, hyperparams