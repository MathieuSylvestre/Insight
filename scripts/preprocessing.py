#set up data as a window from start time end-time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import xgboost
import model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
import sakura_utils as sutils

#Cities considered (more to come)
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha']

#Get and store windows
dfs_w = []
for city in Cities:
    #Load city related data and use the city as an index
    df = pd.read_csv('../data/cleaned/' + city + '_daily.csv')
    #store windows
    dfs_w.append(sutils.get_all_windows(df,50,150))
    
#Get permutations for train/test split, where two cities are used for training and
#others are used for testing to ensure train/test separation despite overlap in windows used
nb_cities = len(Cities)
train = np.zeros(nb_cities,dtype='int')
train[0]=1
train[1]=1
train=np.random.permutation(train)

train_df = []
test_df = []
for i in range(nb_cities):
    if train[i] == 0:
        train_df.append(dfs_w[i])
    else:
        test_df.append(dfs_w[i])

train_df = pd.concat(train_df)
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = pd.concat(test_df)
test_df = test_df.sample(frac=1).reset_index(drop=True)


#Do k-fold cross validation for model selection
nb_cities = len(Cities)
train = np.zeros(nb_cities,dtype='int')
train[0]=1
train[1]=1
train=np.random.permutation(train)

train_df = []
test_df = []
for i in range(nb_cities):
    if train[i] == 0:
        train_df.append(dfs_w[i])
    else:
        test_df.append(dfs_w[i])

#all training samples
train_all_df = pd.concat(train_df)
test_all_df = pd.concat(test_df)

#Define model
xgbr = xgboost.XGBRegressor(max_depth=8,n_estimators = 500, 
                            objective = 'reg:squarederror')

#Return metrics related to cross validation by city
maes, r2s = sutils.cross_validate_by_city(df = train_df,
                                          mdl = xgbr,
                                          target_col='Time_To_Peak',
                                          col_to_drop=['Date'])

print('MAEs and R^2:')
print(maes)
print(r2s)

#Set up for model and normalize as per common practice

train_all_df = train_all_df.sample(frac=1).reset_index(drop=True)

y_train = train_all_df.Time_To_Peak.values
x_train = train_all_df.drop(columns = ['Date','Time_To_Peak']).values

#No need to shuffle test set. Not suffling facilitates plotting predictions
#elegantly 
y_test = test_all_df.Time_To_Peak.values
x_test = test_all_df.drop(columns = ['Date','Time_To_Peak']).values
#normalize
min_max_scaler = preprocessing.MinMaxScaler()
xn_train = min_max_scaler.fit_transform(x_train)
xn_test = min_max_scaler.transform(x_test)

##train model
#regr = model.WindowedRegressionModel(regr = 'Random Forest Regressor', max_depth=8, random_state=0, n_estimators=250)
#regr.train(xn_train,y_train)
#
#y_pred = regr.predict(xn_test)
#print('With Random Forrest')
#print('MAE : ' + str(regr.get_mae(y_test,y_pred)))
#print('R^2 : ' + str(regr.get_r2(y_test,y_pred)))

print('With XGBRegressor')
xgbr.fit(xn_train,y_train)
print('Test set predictions')
y_pred = xgbr.predict(xn_test)
print('MAE : ' + str(median_absolute_error(y_test,y_pred)))
print('R^2 : ' + str(r2_score(y_test,y_pred)))

#Plot predictions over time for the last year of a city from the test set
points_to_plot = 40#+152
dist_from_end = 0 #+152
if dist_from_end < 1:
    sutils.plot_predictions_over_time(y_test[-points_to_plot:],
                                      y_pred[-points_to_plot:])
else:
    sutils.plot_predictions_over_time(y_test[-points_to_plot:-dist_from_end],
                                      y_pred[-points_to_plot:-dist_from_end])

#sutils.plot_predictions_over_time(y_test,y_pred) #too much
print('Train set predictions')
y_pred = xgbr.predict(xn_train)
print('MAE : ' + str(median_absolute_error(y_test,y_pred)))
print('R^2 : ' + str(r2_score(y_test,y_pred)))