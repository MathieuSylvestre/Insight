import pandas as pd
import numpy as np
import time
from datetime import datetime

#Cities considered
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha','washington']

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

#Create dictionary for precipitation descriptions
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

#load target dates
df_target = pd.read_csv('../data/raw/peak_bloom_all.csv')

aggregate_into_days = True

#To aggregate daily data (there are 4 data points of each type (temp, humidity, etc. each day)
def aggregate_mean(x1,x2,x3,x4):
    return np.mean([x1,x2,x3,x4])

def aggregate_min(x1,x2,x3,x4):
    return np.min([x1,x2,x3,x4])

def aggregate_sum(x1,x2,x3,x4):
    return np.sum([x1,x2,x3,x4])

def date_to_nth_day(date, str_format='%Y-%m-%d'):
    """
    Returns the number of days since August 1 from a string in the form 
    'YYYY-MM-DD'
    """
    date = datetime.strptime(date, str_format)
    new_year_day = datetime(year=date.year, month=1, day=1)
    nb_days = (date - new_year_day).days + 1
    if nb_days > 212:        
        return nb_days + - 211 #153 is offset for August 1
    return (date - new_year_day).days + 365 - 211#return actual day of year otherwise

def quantify_description(df, str_desc, col_desc, col, val_true):
    """
    Quantifies descriptors (i.e. precipitation). Takes in description and column 
    containing description (col), returns updated value for col, i.e. modifies 
    the column 'col' of dataframe df
    """
    #Get boolean about whether the string is contained in column
    ind = df[col_desc].str.contains(str_desc,case=True, regex=False)
    #Get array of value to add to each row of col based on value contained
    update_in_vals = np.where(ind, val_true, 0)
    old_vals = df[col].to_numpy()
    updated_vals = old_vals + update_in_vals
    df[col] = updated_vals
    
#To record time
start_time = time.time()

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

        #Add quantitative description
        for description in Prec_Descriptions:     
            val = Prec_Descriptions[description]
            quantify_description(df, description, 'Desc' + str(i+1), 'Prec' + str(i+1), val) 
    
    df.drop(columns = ['Desc1','Desc2','Desc3','Desc4'],inplace=True)   
    
    #interpolate missing information
    df.interpolate(limit = 30, inplace = True)
        
    #Set targets and time since last peak as a feature
    targets = df_target[city].to_list()
    
    #remove Nans from targets list
    targets = [t for t in targets if str(t) != 'nan']
    df['Time_Since_Peak']= np.nan
    df['Is_Peak_Bloom']= 0 
    df['Time_To_Peak']= np.nan
    
    for target in targets:
        
        #reformat if required (revome 0 if day of month is 08 instead of 8) to enable matching
        if target[8] == '0': 
            target = target[:8] + target[-1]
           
        #indicator of peak day
        df.loc[df.Date == target, ['Time_Since_Peak','Is_Peak_Bloom','Time_To_Peak']] = 0, 1, 0
        
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
        df['Time_Since_Peak'] = df['Time_Since_Peak'].interpolate() #Insert values for all dates
        
    #Remove initial Nans
    if len(df.loc[pd.isna(df['Temp1']), :].index.tolist()) > 0:
        max_Nan_index = df.loc[pd.isna(df['Temp1']), :].index.tolist()[-1]
        df = df[max_Nan_index+1:]
    
    if aggregate_into_days:
        #aggregate weather data into single daily value for each feature
        df['Temp'] = df.apply(lambda x: aggregate_mean(x.Temp1, x.Temp2, x.Temp3, x.Temp4), axis=1)     
        df['Temp_Low'] = df.apply(lambda x: aggregate_min(x.Temp_Low1, x.Temp_Low2, x.Temp_Low3, x.Temp_Low4), axis=1)
        df['Baro'] = df.apply(lambda x: aggregate_mean(x.Baro1, x.Baro2, x.Baro3, x.Baro4), axis=1)
        #df['Wind'] = df.apply(lambda x: aggregate_mean(x.Wind1, x.Wind2, x.Wind3, x.Wind4), axis=1)
        df['Hum'] = df.apply(lambda x: aggregate_mean(x.Hum1, x.Hum2, x.Hum3, x.Hum4), axis=1)
        df['Prec'] = df.apply(lambda x: aggregate_sum(x.Prec1, x.Prec2, x.Prec3, x.Prec4), axis=1)

        #Drop specific data that has been aggregated into others
        df.drop(columns = ['Temp1','Temp2','Temp3','Temp4',
                           'Temp_Low1','Temp_Low2','Temp_Low3','Temp_Low4',
                           'Baro1', 'Baro2', 'Baro3', 'Baro4',
                           'Wind1','Wind2','Wind3','Wind4',
                           'Hum1','Hum2','Hum3','Hum4',
                           'Prec1','Prec2','Prec3','Prec4',],inplace=True)
        
    #add Latitude
    df['Latitude'] = df_latitudes.loc[city, 'Latitude']
    
    #add day of year
    df['Day_Of_Year'] = df.apply(lambda x: date_to_nth_day(x.Date), axis=1) 
    
    #drop time points before or after first and latest sakura
    i_to_drop = df[df['Time_To_Peak'] < 0 ].index #Sakura hasn't occured yet
    df.drop(i_to_drop , inplace=True)     
    i_to_drop = df[df['Time_Since_Peak'] < 0 ].index #Latest date of Sakura unknown
    df.drop(i_to_drop , inplace=True)           
    
    #save to CSV as cleaned data
    df.to_csv(r'../data/cleaned/' + city + '_daily.csv',index=False)


