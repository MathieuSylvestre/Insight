# Bloom Cast

Project performed as part of the Insight Artificial Intelligence in Toronto where the goal is to predict when the sakura will occur in different cities around the world. Specifically, at any given date, for a given city an estimate for the number of days before the peak bloom of the sakura is provided. Having a good prediction enables sakura festival organizers to better organize and plan their festivals, which could result in increased number of visitors and revenue. For a brief presentation on the work performed, see [this presentation](https://tinyurl.com/y37emx5z).

### Contents
* [Installation](#installation)
* [Download Data](#download-data)
* [Survival Analysis](#survival-analysis)
* [XGBoost](#xgboost)
* [Predictions](#predictions)
* [Links](#links)

##  Installation

If you have `conda` installed, then create a new virtual environment (using python 3.6+, compatibility issues were not tested for in lower versions) and activate the virtual environment:  
```
conda create -n bloomcast python=3.6
conda activate bloomcast
```

or if you use the normal python virtual environment package `venv` then run:  
```
python3 -m venv bloomcast
source bloomcast/bin/activate
```
Then clone the repo and enter the directory:  
```
git clone https://github.com/MathieuSylvestre/Insight
cd Insight
```

Install the packages from `requirements.txt`:  
```
pip install -r requirements.txt
```

## Download Data

The raw and cleaned data used for training can be found in the `data` folder. To download the weather data from its source, run `notebook/weather_scraper.ipynb`. For predictions, download only the cities of interest, for the latest year. Different cities can also be included as long as historic weather for the past 100 days are available on [this site](https://www.timeanddate.com/weather/). Historical peak bloom dates and geographic information used for training can be found in the `tables` folder. New locations and the corresponding geographical information should be added to these files for predictions at new locations. After making these adjustments, run `scripts/cleaning.py`. This will produce a cleaned dataset, saved as a `.csv` file.

## Survival Analysis

The file `script/ts_cox.py` contains the workflow used to obtain the cox proportional hazards model. To train a new model, select the cities for which data has been downloaded and cleaned with `scripts/cleaning.py`. The file also produces plots illustrating the model's predictions for one sample, from which additional information such as the probability of occurrence of peak bloom over any interval can be computed.

## XGBoost

Following cross-validation, a gradient-boosted tree was trained using the optimal hyperparameters and the final model is saved as `scripts/model.sav`. The file `scripts/gradient_boosting.py` contains the workflow used for training the xgboost model. The following plot illustrates the performance of the model as a function of the date to peak bloom.

![Median Absolute Error as a Function of the Number of Days to Peak Bloom](/images/readme_img.png)

## Predictions 

New predictions can be made with `scripts/make_predictions.py`. The code can be easily modified to include cities not used for training, though the city's latitude must be provided. For better results with a new city, it may be preferable to get more relevant data, especially if its climate/geography is sufficiently different from the cities used for training. See [Download Data](#download-data) for details.

## Links

[Slides](https://tinyurl.com/y37emx5z) 