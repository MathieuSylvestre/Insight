import numpy as np
import xgboost
from sklearn.metrics import median_absolute_error
import sakura_utils as sutils
import pickle

#Define list of cities considered
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha','washington']
nb_cities = len(Cities)

#Define hyper-parameters to get appropriate windows
window_length = 100
max_distance_to_peak = 150

#Get windows
dfs_w = sutils.get_lists_of_windows(window_length, 
                                    max_distance_to_peak, 
                                    Cities, 
                                    path = '../data/cleaned/',
                                    col_to_drop = ['Hum','Prec','Baro','Temp_Low'])
                                    #col_to_drop = ['Hum','Prec','Baro','Temp_Low'])

#Get train and test sets for windows. Separation by city and year is performed 
#(i.e. all windows from any given city and year are grouped together and assigned
#to the training or test set) to counteract violation of IID due to window 
#construction methodology
train_df, test_df = sutils.get_train_test_split_by_date_and_city(dfs_w,test_size=0.2)

#get data for model training
xn_train, y_train, xn_test, y_test = sutils.prepare_data_for_training(train_df,
                                                   test_df,
                                                   drop_Time_Since_Peak = True, 
                                                   drop_Day_Of_Year = False, 
                                                   drop_Latitude = False,
                                                   return_numpy=True)

#Weigh samples. Smaller weights are assigned to samples far from the actual
#bloom since we assume that required information to make a better prediction 
#isn't yet available (i.e. the weather of the future is necessary)
#To weigh samples, a function that takes in samples and returns the samples
#weights is defined. It can be sent as a parameter to certain functions is 
#sakura_utils 
days_before_offset = 30 #must be at least 1
def get_weighting_function(scaling_factor, days_before_offset):
    def weight(y_train):
        dividers = np.max((days_before_offset*np.ones(len(y_train)),
                   y_train),axis=0)-(days_before_offset-1)*np.ones(len(y_train))
        return scaling_factor*np.ones(len(y_train))/dividers
    return weight

# define hyperparameters over which to perform grid search, then perform grid search
depths = [i for i in range(5,11)]
estimators = [25*i for i in range(4,15)]
weighting_function = get_weighting_function(100,30)
#avg_maes, hyperparams = sutils.grid_search_cross_validation_XGBR(train_df, depths = depths, estimators = estimators, weighting = weighting_function, target_col = 'Time_To_Peak', col_to_drop = ['Date', 'Time_Since_Peak'], verbose = True)

#Train model with all training data
xgbr = xgboost.XGBRegressor(max_depth=7,
                            n_estimators = 200,
                            objective = 'reg:squarederror')

xgbr.fit(xn_train,y_train,weighting_function(y_train))

#Save model for future retrieval
filename = 'model' + '.sav'
pickle.dump(xgbr, open(filename, 'wb'))

#make predicitions
print('Test set predictions')
y_pred = xgbr.predict(xn_test)
print('MAE : ' + str(median_absolute_error(y_test,y_pred)))

#Plot predictions over time for the last year of a city from the test set
points_to_plot = 50#+2*152
dist_from_end = 0#+2*152
if dist_from_end < 1:
    sutils.plot_predictions_over_time(y_test[-points_to_plot:],
                                      y_pred[-points_to_plot:])
else:
    sutils.plot_predictions_over_time(y_test[-points_to_plot:-dist_from_end],
                                      y_pred[-points_to_plot:-dist_from_end])

#Plot true vs predicted for some year
sutils.plot_predictions_versus_true(y_test,y_pred,100,1)

#show training set predictions
print('Training set predictions')
y_pred_train = xgbr.predict(xn_train)
print('MAE : ' + str(median_absolute_error(y_train,y_pred_train)))

#Plot median and average errors for both the training and test set
sutils.plot_median_error_vs_y_true(y_pred,y_test,60)
mae_test = sutils.plot_mae_vs_y_true(y_pred,y_test,150)
sutils.plot_median_error_vs_y_true(y_pred_train,y_train,150)
sutils.plot_mae_vs_y_true(y_pred_train,y_train,150)
    