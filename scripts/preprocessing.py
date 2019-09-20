import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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
#descriptions = []

#load target dates
df_target = pd.read_csv('../data/raw/peak_bloom_japan.csv')

#To aggregate daily data
def aggregate_mean(x1,x2,x3,x4):
    return np.mean([x1,x2,x3,x4])

def aggregate_min(x1,x2,x3,x4):
    return np.min([x1,x2,x3,x4])

#for MVP
for city in Cities:
    df = pd.read_csv('../data/raw/weather_all/' + city + '.csv',header=None)
    
    #df = df.dropna()    
    df.columns = column_names
#        print('\n\n' + city + '\n')
#        print(df.loc[pd.isna(df['Temp1']), :].index.tolist())

    df.drop(columns = ['Desc1','Desc2','Desc3','Desc4'],inplace=True)
    df.drop(columns = ['Wind_D1','Wind_D2','Wind_D3','Wind_D4'],inplace=True)
    
    #interpolate missing information
    df.interpolate(limit = 30, inplace = True)
    df.loc[pd.isna(df['Temp1']), :].index.tolist()
        
    #Set targets
    targets = df_target[city].to_list()
    df['Is_Peak_Bloom']= 0 
    df['Time_To_Peak']= np.nan
    for target in targets:
        
        #reformat if required
        if target[8] == '0': 
            target = target[:8] + target[-1]
            
        df.loc[df.Date == target, ['Is_Peak_Bloom','Time_To_Peak']] = 1, 0
        #here is where I may add ones around the peak bloom to help balance the dataset
        
    #Set Time_to_Peak for regression
    
    peak_indices = df.index[df.Is_Peak_Bloom == 1].tolist()
    
    #print(peak_indices)
    if len(peak_indices)>0:
        if len(peak_indices)>1:
            for i in range(len(peak_indices)-1):
                df['Time_To_Peak'].iloc[peak_indices[i]+1] = peak_indices[i+1]-peak_indices[i]-1
        df['Time_To_Peak'].iloc[0] = peak_indices[0]
        df['Time_To_Peak'].iloc[-1] = -1000
        df['Time_To_Peak'] = df['Time_To_Peak'].interpolate() #Set targest for regression
        
    #remove initial Nans
    if len(df.loc[pd.isna(df['Temp1']), :].index.tolist()) > 0:
        max_Nan_index = df.loc[pd.isna(df['Temp1']), :].index.tolist()[-1]
        df = df[max_Nan_index+1:]
    
    #aggregate weather data into single daily value for each feature
    df['Temp'] = df.apply(lambda x: aggregate_mean(x.Temp1, x.Temp2, x.Temp3, x.Temp4), axis=1)
    
    
    df['Temp_Low'] = df.apply(lambda x: aggregate_min(x.Temp_Low1, x.Temp_Low2, x.Temp_Low3, x.Temp_Low4), axis=1)
    #COMMENTED OUT FOR NOW
    df['Baro'] = df.apply(lambda x: aggregate_mean(x.Baro1, x.Baro2, x.Baro3, x.Baro4), axis=1)
    #df['Wind'] = df.apply(lambda x: aggregate_mean(x.Wind1, x.Wind2, x.Wind3, x.Wind4), axis=1)
    df['Hum'] = df.apply(lambda x: aggregate_mean(x.Hum1, x.Hum2, x.Hum3, x.Hum4), axis=1)

    #Drop specific data
    df.drop(columns = ['Temp1','Temp2','Temp3','Temp4',
                       'Temp_Low1','Temp_Low2','Temp_Low3','Temp_Low4',
                       'Baro1', 'Baro2', 'Baro3', 'Baro4',
                       'Wind1','Wind2','Wind3','Wind4',
                       'Hum1','Hum2','Hum3','Hum4'],inplace=True)
    
    #Assemble data as windows that can be shifted for MVP
    window_length = 50
    
    df_windows = df.drop(columns = ['Date','Is_Peak_Bloom','Time_To_Peak'],inplace=False)
    for i in range(1,window_length):
        df_temp = df.shift(-i)
        df_temp.drop(columns = ['Date','Is_Peak_Bloom','Time_To_Peak'],inplace=True)
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
    
    #append Latitude as a feature
    df_windows['Latitude'] = df_latitudes.loc[city, 'Latitude']
    
    dfs.append(df_windows)


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



#x_train_normalized, y_train, x_test_normalized, y_test
