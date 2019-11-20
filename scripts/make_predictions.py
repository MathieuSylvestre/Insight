import joblib
from datetime import datetime , timedelta
from urllib.request import Request, urlopen
import time
import calendar
from dateutil.relativedelta import relativedelta
import pandas as pd

#Define city and latitude where predictions are to be made
city = 'japan/nagasaki'
df_latitudes = pd.read_csv('../data/raw/city_geo_data.csv')
df_latitudes = df_latitudes.set_index('City')
latitude = df_latitudes.loc[city.split('/')[-1], 'Latitude']

def scrape_weather(begin_date, end_date, Measurements, city):
    """
    Get weather data from a location over a range of dates
    
    Parameters
    ----------
    begin_date: beginning of date range, string of format "YYYY-MM-DD"
    end_date: end of date range, string of format "YYYY-MM-DD"
    Measurements: list of measurements to extract. Default is all available measurements,
        i.e. Measurements = ['\"desc\"','\"temp\"','\"templow\"','\"baro\"','\"wind\"','\"wd\"','\"hum\"']
    city: string, location of city, as required in the url of timeanddate (typically '<country>/<city>).
        See www.timeanddate.com/weather/ for details
        
    Returns
    -------
    weather_data: list of lists containing the date and raw data for the given city over the given 
        range of dates
    """
    current_date = datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")+relativedelta(months=+1)
        #add one to make loop end on the end_date
    
    #get website from which to scrape weather data
    lookup_URL = 'https://www.timeanddate.com/weather/{}/historic?month={}&year={}'
    
    #List to store data
    weather_data = []
    
    #Cycle through every date to extract .html
    while current_date <= end_date:
        
        formatted_lookup_URL = lookup_URL.format(city,
                                                 current_date.month,
                                                 current_date.year)
        str_month = current_date.strftime("%B")
        #print(formatted_lookup_URL) #to track progress
        
        try:
            req = Request(formatted_lookup_URL, headers={'User-Agent': 'Mozilla/5.0'})
            str_html = urlopen(req).read().decode('utf-8')
            
            for i in range(1,calendar.monthrange(current_date.year,current_date.month)[1]+1):
                index = str_html.find(str_month + ' ' + str(i) + ',')
                str_html = str_html[index:]
                day_data = [current_date.strftime("%Y-%m") + '-' + str(i)]
                
                #for each day, there are 4 time periods
                for j in range(4):
                    for measurement in Measurements:
                        index = str_html.find(measurement)
                        str_html = str_html[index:]
                        index = str_html.find(':')
                        str_html = str_html[index+1:]
                        if measurement == '\"hum\"' :
                            index = str_html.find('}')
                        else:
                            index = str_html.find(',')
                        day_data.append(str_html[:index])
                weather_data.append(day_data)
                    
            current_date += relativedelta(months=+1) 
            
        except:
            #Wait, too many requests were made
            time.sleep(1)
            
    return weather_data

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

def make_prediction_today(city, latitude):
    
    Measurements = ['\"desc\"','\"temp\"','\"templow\"','\"baro\"','\"wind\"','\"wd\"','\"hum\"']
    begin_date = datetime.strftime(datetime.now() - timedelta(150), '%Y-%m-%d')#extra days in case of missing data
    end_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')#'2019-11-18'
    
    #get weather data as a dataframe
    weather_data = scrape_weather(begin_date, end_date, Measurements, city)
    
    #remove data that has yet to be uploaded
    for i in range(len(weather_data)-1,100,-1):
        if weather_data[i][-1] == '':
            weather_data.pop()
        else:
            break
    
    #only keep last 100 elements, since our window is of length 100
    weather_data = weather_data[-101:]
    
    #Only keep temperature
    weather_needed = []
    avg_temp = 15 #dummy value, only used if value is missing 
    for day in weather_data:
        date = day[0]
        #suppose same avg_temp as previous day if missing data
        try:
            avg_temp = (float(day[2]) + float(day[9]) + float(day[16]) + float(day[23]))/4
        except:
            pass #suppose same avg_temp as previous day if missing data
        weather_needed.append([date,avg_temp])
    
    #make a dataframe to facilitate manipulation    
    df = pd.DataFrame(weather_needed)
    df.columns = ['Date','Temp']
    
    #add Latitude
    df['Latitude'] = latitude
    #add day of year
    df['Day_Of_Year'] = df.apply(lambda x: date_to_nth_day(x.Date), axis=1) 
    #drop date
    df.drop(columns = ['Date'],inplace=True)
    
    df_windows = df.drop(columns = ['Day_Of_Year','Latitude'])
    for i in range(1,100):
        df_temp = df.shift(-i)
        df_temp.drop(columns = ['Day_Of_Year','Latitude'],inplace=True)
        df_windows = pd.concat([df_windows,df_temp],axis=1)
    #include target and latitude on last frame
    df_windows = pd.concat([df_windows,df.shift(-100)],axis=1)
    df_windows = df_windows.dropna() #Only keep latest window, all others have nans due to shifting
    
    x = df_windows.values #numpy x is required for predictions
    
    #load model and normalizing function
    loaded_model = joblib.load("model")
    loaded_min_max_scaler = joblib.load("min_max_scaler")
    xn = loaded_min_max_scaler.transform(x)
    
    #make prediction
    date_of_prediction = weather_data[-1][0]
    predicted_days_to_bloom = loaded_model.predict(xn) #in days from date_of_prediction
    
    return datetime.strftime(datetime.strptime(date_of_prediction,"%Y-%m-%d") + timedelta(int(predicted_days_to_bloom[0])),"%Y-%m-%d"), date_of_prediction

#make predictions
prediction, date_of_prediction = make_prediction_today(city,latitude)

#Print prediction
print('Prediction for ' + city.split('/')[-1] + ' made on ' + date_of_prediction + ' : ' + prediction)