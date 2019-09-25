import pandas as pd

#Cities considered (more to come)
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha']

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

#Load target dates 
df_target = pd.read_csv('../data/raw/peak_bloom_japan.csv')

#Get baseline as average of all previous years
baseline = []
mad= []
for city in Cities:
    df_dayofyear = pd.to_datetime(df_target[city]).dt.dayofyear
    avg_bloom = df_dayofyear.mean(skipna=True)
    baseline.append(avg_bloom)
    mad.append(df_dayofyear.mad(skipna=True))
    
baselineMAE = sum(mad)/len(mad)
    