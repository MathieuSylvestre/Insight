import numpy as np
from lifelines import CoxPHFitter
import sakura_utils as sutils
import matplotlib
from matplotlib import pyplot as plt

plot_result = False #Define whether a sample output plot should be produced

#Cities considered
Cities = ['sapporo','niigata','aomori','kanazawa','hiroshima','sendai', 
          'kyoto', 'tokyo', 'fukuoka', 'shizuoka','matsuyama','osaka','nagoya',
          'nagasaki','kagoshima','naha','washington']

#Do k-fold cross validation for model selection
nb_cities = len(Cities)

#Assign cities to train and test cities
window_length = 100
max_distance_to_peak = 150

dfs_w = sutils.get_lists_of_windows(window_length, 
                                    max_distance_to_peak, 
                                    Cities,
                                    path = '../data/cleaned/',
                                    col_to_drop = ['Temp_Low'])#['Hum','Prec','Baro','Temp_Low'])


#get training and test data
train_df, test_df  = sutils.get_train_test_split_by_date_and_city(dfs_w,test_size=0.2)

#get data as a pandas dataframe for coxPHFitter
weight_delays = 30
n_train, n_test = sutils.prepare_data_for_training(train_df,
                                                   test_df,
                                                   drop_Time_Since_Peak = True, 
                                                   drop_Day_Of_Year = False, 
                                                   drop_Latitude = False,
                                                   return_numpy=False,
                                                   weight_delays = weight_delays)

#Create and train Cox Proportional Hazards model
cph = CoxPHFitter()
if weight_delays != None:
    cph.fit(n_train, duration_col='Target', weights_col = 'Weights', show_progress=True)
else:
    cph.fit(n_train, duration_col='Target', show_progress=True)
cph.print_summary()

#Quantify predictions
y_test = n_test.Target.values
y_pred = cph.predict_median(n_test).values.T[0]
#get mean average error for last 100 days
mae_test = sutils.plot_mae_vs_y_true(y_pred,y_test,100)

#Plot results for presentation slides
if plot_result:
    
    #Define times at which to plot CDF
    times = np.arange(140).tolist()
    
    #Plot lines delimiting an interval in time and corresponding values of the CDF
    low_x = 60
    high_x = 80
    low_y = 1 - cph.predict_survival_function(n_test[70:71], [low_x])
    high_y = 1 - cph.predict_survival_function(n_test[70:71], [high_x])
    plt.figure()
    plt.figure(figsize=(6,6))
    plt.plot([low_x,low_x],[0,low_y],color="purple")
    plt.plot([high_x,high_x],[0,high_y],color="purple")
    plt.plot([0,low_x],[low_y,low_y],color="purple")
    plt.plot([0,high_x],[high_y,high_y],color="purple")
    
    #Get x and y at the 0.5 mark (cph.predict_median returns an integer)
    x_med = cph.predict_median(n_test[70:71]) + 1 #start lower to get close
    y_med = cph.predict_survival_function(n_test[70:71],times = [x_med])
    while y_med < 0.5:  
        x_med -= 0.02
        y_med = cph.predict_survival_function(n_test[70:71],times = [x_med])
    
    plt.plot([x_med,x_med],[0,0.5],color="purple")
    plt.plot([0,x_med],[0.5,0.5],color="purple")
    
    #Plot CDF, set appropriate ticks and label axes
    plt.plot(np.ones(len(times))-cph.predict_survival_function(n_test[70:71], times=times).values.T)
    matplotlib.pyplot.yticks([0,0.2,0.4,0.6,0.8,1.0])
    xticks= np.arange(0, 150, step=20).tolist()
    matplotlib.pyplot.xticks(xticks)
    matplotlib.rcParams.update({'font.size': 16})
    plt.ylabel(r'P(Peak bloom has arrived)')
    plt.xlabel(r'Days from today')