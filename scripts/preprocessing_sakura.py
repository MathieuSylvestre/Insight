import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Masking
from keras.callbacks import EarlyStopping

import model

#Cities considered (more to come)
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha']

#Load city related data and use the city as an index
df_latitudes = pd.read_csv('../data/raw/city_geo_data.csv')
df_latitudes = df_latitudes.set_index('City')

#Set column names - there are 4 weather measurements per day
column_names = ['Date']
features = ['Desc','Temp','Temp_Low','Baro','Wind','Wind_D','Hum'] 
for i in range(4):
    for feature in features:
        feat = feature + str(i+1)
        column_names.append(feat)

dfs = []

##Exploratory analysis: get all weather descriptions
#descrip = []
#for city in Cities:
#    df = pd.read_csv('../data/raw/weather_all/' + city + '.csv',header=None)
#    
#    #df = df.dropna()    
#    df.columns = column_names
#    
#    descrip.append(df.Desc1.unique())
#    descrip.append(df.Desc2.unique())
#    descrip.append(df.Desc3.unique())
#    descrip.append(df.Desc4.unique())
#
##Get set of all names
#descriptions = set(x for l in descrip for x in l)

#Create dictionary for descriptions
Prec_Descriptions = {
    'Drizzle.': 0.1,
    'Heavy rain.':1,
    'Light rain.': 0.1,
    'Light mixture of precip.': 0.1,
    'Light snow.': 0.1,
    'Lots of rain.': 0.8,
    'Rain showers.': 0.3,
    'Rain.': 0.5,
    'Scattered showers.': 0.15,
    'Snow flurries': 0.05,
    'Snow showers': 0.1,
    'Snow':0.2,
    'Heavy snow.':0.6,
    'Sprinkles.': 0.1,
    'Strong thunderstorms.':0.3,
    'Thunderstorms.': 0.1,
    'Thundershowers.': 0.3,
}

Sun_Descriptions = {
#    'Low clouds.': 0,
#    'Cloudy.':0
    'Broken clouds.':0.3,
    'Mostly cloudy.':0.15,
    'Overcast.':0.05,
    'Partly sunny.': 0.5,
    'Partly cloudy.':0.5,
    'Passing clouds.':0.7,
    'Fog.': 0.2,
    'Dense fog.':0.05,
    'Clear.':0.95,
    'More clouds than sun.':0.3,
    'Scattered clouds.':0.7,
}

#load target dates
df_target = pd.read_csv('../data/raw/peak_bloom_japan.csv')

aggregate_into_days = True
window_length = None

#To aggregate daily data
def aggregate_mean(x1,x2,x3,x4):
    return np.mean([x1,x2,x3,x4])

def aggregate_min(x1,x2,x3,x4):
    return np.min([x1,x2,x3,x4])

def aggregate_sum(x1,x2,x3,x4):
    return np.sum([x1,x2,x3,x4])

#Take in description and column containing description, return updated value for column 'col'. col should be a string
#Modifies the column col of dataframe at the 
def quantify_description(df, str_desc, col_desc, col, val_true):
    #if description contains string in each of the following,
    
    #Get boolean about whether the string is contained in column
    ind = df[col_desc].str.contains(str_desc,case=True, regex=False)
    #Get array of value to add to each row of col based on value contained
    update_in_vals = np.where(ind, val_true, 0)
    old_vals = df[col].to_numpy()
    updated_vals = old_vals + update_in_vals
    df[col] = updated_vals
    
#To record time
start_time = time.time()

#for MVP
for city in Cities:
    df = pd.read_csv('../data/raw/weather_all/' + city + '.csv',header=None)
    
    #Set column names to facilitate manipulations
    df.columns = column_names
    
    #Drop wind direction, assume it isn't relevant between cities
    df.drop(columns = ['Wind_D1','Wind_D2','Wind_D3','Wind_D4'],inplace=True)
    
    #Replace Nans in descriptions by empty string and create columns that quantify descriptions
    for i in range(4):
        df['Desc' + str(i+1)].fillna('', inplace=True)
        df['Prec' + str(i+1)] = 0
#        df['Sun'  + str(i+1)] = 0

        #Add quantitative description
        for description in Prec_Descriptions:     
            val = Prec_Descriptions[description]
            quantify_description(df, description, 'Desc' + str(i+1), 'Prec' + str(i+1), val) 
        
#        for description in Sun_Descriptions:    
#            val = Sun_Descriptions[description]
#            quantify_description(df, description, 'Desc' + str(i+1), 'Sun'  + str(i+1), val)              

#    print(df[['Desc1','Prec1']])
#    print(df[['Desc2','Prec2']])
    
    df.drop(columns = ['Desc1','Desc2','Desc3','Desc4'],inplace=True)   
    
    #interpolate missing information
    df.interpolate(limit = 30, inplace = True)
    df.loc[pd.isna(df['Temp1']), :].index.tolist()
        
    #Set targets and time since last peak as a feature
    targets = df_target[city].to_list()
    
    #remove Nans from targets list
    targets = [t for t in targets if str(t) != 'nan']
    df['Time_Since_Peak']= np.nan
    df['Is_Peak_Bloom']= 0 
    df['Time_To_Peak']= np.nan

    print('Targets of ' + city + ' :')
    print(targets)
    
    for target in targets:
        
        #reformat if required (revome 0 if day of month is 08 instead of 8) to enable matching
        if target[8] == '0': 
            target = target[:8] + target[-1]
            
        df.loc[df.Date == target, ['Time_Since_Peak','Is_Peak_Bloom','Time_To_Peak']] = 0, 1, 0
        #here is where I may add ones around the peak bloom to help balance the dataset
        
    #Set Time_to_Peak for regression and Time_Since_Peak as a feature
    peak_indices = df.index[df.Is_Peak_Bloom == 1].tolist()

    if len(peak_indices)>0:
        if len(peak_indices)>1:
            for i in range(1,len(peak_indices)-1):
                df['Time_To_Peak'].iloc[peak_indices[i]+1] = peak_indices[i+1]-peak_indices[i]-1
                df['Time_Since_Peak'].iloc[peak_indices[i]-1] = peak_indices[i]-peak_indices[i-1]-1
            #Add first for Time_To_Peak
            df['Time_To_Peak'].iloc[peak_indices[0]+1] = peak_indices[1]-peak_indices[0]-1
            df['Time_Since_Peak'].iloc[peak_indices[-1]-1] = peak_indices[-1]-peak_indices[-2]-1
        
        df['Time_To_Peak'].iloc[0] = peak_indices[0]
        df['Time_Since_Peak'].iloc[-1] = len(df)-peak_indices[-1]-1
        
        #Set dummy values to rows that will be deleted after further preprocessing
        df['Time_To_Peak'].iloc[-1] = -1000
        df['Time_Since_Peak'].iloc[0] = -1000
        
        df['Time_To_Peak'] = df['Time_To_Peak'].interpolate() #Set targest for regression
        df['Time_Since_Peak'] = df['Time_Since_Peak'].interpolate() #Set missing data
        
    #Remove initial Nans
    if len(df.loc[pd.isna(df['Temp1']), :].index.tolist()) > 0:
        max_Nan_index = df.loc[pd.isna(df['Temp1']), :].index.tolist()[-1]
        df = df[max_Nan_index+1:]
    
    if aggregate_into_days:
        #aggregate weather data into single daily value for each feature
        df['Temp'] = df.apply(lambda x: aggregate_mean(x.Temp1, x.Temp2, x.Temp3, x.Temp4), axis=1)     
        df['Temp_Low'] = df.apply(lambda x: aggregate_min(x.Temp_Low1, x.Temp_Low2, x.Temp_Low3, x.Temp_Low4), axis=1)
        #COMMENTED OUT FOR NOW
        df['Baro'] = df.apply(lambda x: aggregate_mean(x.Baro1, x.Baro2, x.Baro3, x.Baro4), axis=1)
        #df['Wind'] = df.apply(lambda x: aggregate_mean(x.Wind1, x.Wind2, x.Wind3, x.Wind4), axis=1)
        df['Hum'] = df.apply(lambda x: aggregate_mean(x.Hum1, x.Hum2, x.Hum3, x.Hum4), axis=1)
        df['Prec'] = df.apply(lambda x: aggregate_sum(x.Prec1, x.Prec2, x.Prec3, x.Prec4), axis=1)
        
        
        #Drop specific data
        df.drop(columns = ['Temp1','Temp2','Temp3','Temp4',
                           'Temp_Low1','Temp_Low2','Temp_Low3','Temp_Low4',
                           'Baro1', 'Baro2', 'Baro3', 'Baro4',
                           'Wind1','Wind2','Wind3','Wind4',
                           'Hum1','Hum2','Hum3','Hum4',
                           'Prec1','Prec2','Prec3','Prec4',],inplace=True)
    
    if window_length != None:
        print('making windows')
#        if history_in_windows:
#            #TODO: add features to model which accounts for  weather
#            pass
        
        #Assemble data as windows that can be shifted for MVP
        df_windows = df.drop(columns = ['Date','Is_Peak_Bloom','Time_Since_Peak','Time_To_Peak'],inplace=False)
        for i in range(1,window_length):
            df_temp = df.shift(-i)
            df_temp.drop(columns = ['Date','Is_Peak_Bloom','Time_Since_Peak','Time_To_Peak'],inplace=True)
            df_windows = pd.concat([df_windows,df_temp],axis=1)
        #include target on last frame
        df_windows = pd.concat([df_windows,df.shift(-window_length)],axis=1)
        
        #only keep windows with target less than 150 days away
        df_windows = df_windows.dropna() #NaNs were generate as a result of the shifting
        # Delete these row indexes from dataFrame    
        i_to_drop = df_windows[df_windows['Time_To_Peak'] > 150 ].index
        df_windows.drop(i_to_drop , inplace=True)
        i_to_drop = df_windows[df_windows['Time_To_Peak'] < 0 ].index #Sakura hasn't occured yet
        df_windows.drop(i_to_drop , inplace=True)     
        i_to_drop = df_windows[df_windows['Time_Since_Peak'] < 0 ].index #Latest date of Sakura unknown
        df_windows.drop(i_to_drop , inplace=True)    
        
        #append Latitude as a feature
        df_windows['Latitude'] = df_latitudes.loc[city, 'Latitude']
        
        dfs.append(df_windows)
        
    else:
        #TODO: setup for time-series analysis
        
        #df.set_index('Date')
        
        print(city + ' - not windowing')
        
        #add Latitude
        df['Latitude'] = df_latitudes.loc[city, 'Latitude']
        #print(df)
        i_to_drop = df[df['Time_To_Peak'] < 0 ].index #Sakura hasn't occured yet
        df.drop(i_to_drop , inplace=True)     
        i_to_drop = df[df['Time_Since_Peak'] < 0 ].index #Latest date of Sakura unknown
        df.drop(i_to_drop , inplace=True)           
        
        #save to CSV as cleaned data
        df.to_csv(r'../data/cleaned/' + city + '_daily.csv',index=False)
        
        dfs.append(df)
        
if window_length != None:
    
    #For windowed data
    df_all = pd.concat(dfs)
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    
    y = df_all.Time_To_Peak.values
    x = df_all.drop(columns = ['Date','Time_To_Peak']).values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train_normalized = min_max_scaler.fit_transform(x_train)
    x_test_normalized = min_max_scaler.transform(x_test)
    
elif False:
    
#    dfs_ts = []
#    x_ts = []
#    y_ts = []
    
    max_len = 0
    for df in dfs:
        max_len = max(max_len, len(df))
    
    #Do zero padding to enable training with various length time series
    
    xs_list = []
    ys_list = []
    
    special_value = -1000
    
    for df in dfs:
        #make block of 0s of dimensions max_len - len(df) by #dimensions
        
        xs1 = df.drop(columns = ['Date','Time_To_Peak']).values
        ys1 = df.Time_To_Peak.values        
        
        #block of zeros to add
        zero_pad = True
        if zero_pad:
            if max_len > len(df):
                zs1 = special_value*np.ones((max_len-len(df),xs1.shape[1]))
                xs1 = np.concatenate((zs1,xs1)) #zero-pad before or after?
                ys1 = np.concatenate((np.zeros(max_len-len(df)),ys1))
        
        xs_list.append(xs1)
        ys_list.append(ys1)

    xs = np.asarray(xs_list)  
    ys = np.asarray(ys_list)  
    
    #reshape output
    ys = ys.reshape(ys.shape[0],ys.shape[1],1)   

     
    
#    model = Sequential()
#    #Runs, doesn't contain much though
#    model.add(LSTM(units = 1, input_shape=(None,xs.shape[-1]),return_sequences=True)) 
#    model.compile(loss='mae', optimizer='adam', metrics=['mean_absolute_error'])
#    model.fit(xs, ys, epochs = 1, batch_size = 4)    
#
#    #Runs!!! add more complexity to the network
#    model = Sequential()
#    model.add(LSTM(units = 64, input_shape=(None,xs.shape[-1]),return_sequences=True))     
#    for i in range(3):
#        model.add(Dense(32, activation='relu'))
#    #model.add(Dropout(0))
#    #Problem, activation needs to be relu for regression here, but could lead to nans
#    model.add(LSTM(units = 1, input_shape = (32,), activation = None, return_sequences=True))     
#    model.compile(loss='mae', optimizer='adam', metrics=['mean_absolute_error'])
#    model.fit(xs, ys, epochs = 1, batch_size = 4)
 
    #Runs, no nans, difference is a simpler model with no Dense layers and less LSTM units
    model = Sequential()
#    model.add(Masking(mask_value=special_value, input_shape=(max_len, xs.shape[-1])))
    model.add(LSTM(units = 12, input_shape=(None,xs.shape[-1]),activation = 'relu', return_sequences=True))     
    model.add(LSTM(units = 1, input_shape = (12,), activation = 'relu', return_sequences=True))     
    model.compile(loss='mae', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(xs, ys, epochs = 1, batch_size = 2)
    
    
end_time = time.time()

print('time = ' + str(end_time - start_time) + ' s')

if False:
    regr = model.WindowedRegressionModel(regr = 'Random Forest Regressor')
    regr.train(x_train,y_train)
    
    y_pred = regr.predict(x_test)
    print('MAE : ' + str(regr.get_mae(y_test,y_pred)))
    print('R^2 : ' + str(regr.get_r2(y = y_test,y_pred = y_pred)))


