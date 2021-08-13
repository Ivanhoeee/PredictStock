# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:53:40 2021

@author: P870306
"""
import numpy as np
import pandas as pd
import sktime
import yfinance as yf
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sktime.forecasting.naive import NaiveForecaster

class Model(): 
    """
    This class is responsible for estimating a model. 
    """
    
    def __init__(self, data_import): 
        self.y_train = data_import.y_train.copy()
        self._y_train_index = self.y_train.index
        self.y_test = data_import.y_test.copy()
        self._y_test_index = self.y_test.index
        
    def create_fc_horizon(self): 
        if len(self.y_test) == 0: 
            self.fh = np.arange(1, 2) 
        else: 
            self.fh = np.arange(1, len(self.y_test) + 1)
        
    def Naive(self): 
        self.model = NaiveForecaster(strategy="last").fit(self.y_train.reset_index(drop = True))
        self.model_name = "NaiveForecaster"
        
    def forecast(self): 
        self.prediction = self.model.predict(self.fh)
        
    def set_fc_index(self, index): 
        """
        This function sets the index of the forecast. If we forecast out-of-sample
        then we don't know the exact dates due to weekend and holidays.
        
        index: datetime index
        """
        self.prediction.index = [index] 
        
    def export_fc(self): 
        """ 
        This function exports the forecast in the JSON format.
        """
        export = {}
        export[self.model_name] = {}
        export[self.model_name]["forecast"] = []
        for i in range(len(self.prediction)):
            day_fc = {}
            day_fc["date"] = self.prediction.index[i].strftime("%Y-%m-%d")
            day_fc["value"] = str(self.prediction.iloc[0])
            export[self.model_name]["forecast"].append(day_fc)
        return export
            
        
        
        

#%% 
model = Model(stock)
model.create_fc_horizon()
model.Naive()
model.forecast()
model.set_fc_index(pd.to_datetime("2021-07-07"))
export = model.export_fc()
        