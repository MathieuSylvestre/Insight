import pandas as pd
import numpy as np

#Cities considered (more to come)
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha','washington']

#Set column names - there are 4 weather measurements per day
column_names = ['Date']
features = ['Desc','Temp','Temp_Low','Baro','Wind','Wind_D','Hum'] 
for i in range(4):
    for feature in features:
        feat = feature + str(i+1)
        column_names.append(feat)

dfs = []

#Get all weather descriptions
descrip = []
for city in Cities:
    df = pd.read_csv('../data/raw/weather_all/' + city + '.csv',header=None)
  
    df.columns = column_names
    
    descrip.append(df.Desc1.unique())
    descrip.append(df.Desc2.unique())
    descrip.append(df.Desc3.unique())
    descrip.append(df.Desc4.unique())

#Get set of all description names
descriptions = set(x for l in descrip for x in l)



#Load target dates to see baseline
df_target = pd.read_csv('../data/raw/peak_bloom_all.csv')

#Get baseline as average of all previous years (different baselines considered)
baseline = []
mean_absolute_deviation = []
previous_average = []

#store all error with respect to historical averages
errors_from_historic_avg = []
errors_from_previous_year = []

for city in Cities:
    df_dayofyear = pd.to_datetime(df_target[city]).dt.dayofyear    
    day_of_year = np.flip(df_dayofyear.values) #Flip to make chronological
    day_of_year = day_of_year[~np.isnan(day_of_year)] #remove nans
    historic_average = day_of_year[0] #initial historic avg is first year
    
    for year in range(1,len(day_of_year)):
        #compute cumulative average in different ways
        errors_from_historic_avg.append(abs(day_of_year[year]-historic_average)) #average, accouting for causaulity
        historic_average = (year * historic_average + day_of_year[year])/(year + 1) #update historic average each year
        errors_from_previous_year.append(abs(day_of_year[year]-day_of_year[year-1])) #
    
    #store another baseline as simple average absolute deviation for a given city 
    mean_absolute_deviation.append(df_dayofyear.mad(skipna=True))

#in the end, set mean absolute deviation as baseline (it is the best)
baseline = np.mean(mean_absolute_deviation)
    