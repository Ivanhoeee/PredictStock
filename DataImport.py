# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:43:19 2021

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

class DataImport():

    def __init__(self):
        self.ticker = None
        self.start = None
        self.end = None
        self.interval = None
        self.group_by = None
        self._orig_data = None
        self.data = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
    
    def import_data(self, ticker, start, end, interval, group_by): 
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.group_by = group_by
        data = yf.download(tickers = self.ticker, 
                           start = self.start, 
                           end = self.end, 
                           interval = self.interval, 
                           group_by = self.group_by, 
                           auto_adjust = True, 
                           prepost = True, 
                           threads = True)
        self._orig_data = data
    
    def prep_data(self, price_type): 
        """
        This function takes in data from yf.download and depending on price_type
        returns a dataframe with the chosen price_type (Open, Close, High) for 
        each ticker. 
    
        Parameters
        ----------
        df : dataframe
            output from yf.download().
        price_type : string
            choose from Close, Open, High.
    
        Returns
        -------
        df: dataframe
            Dataframe where per ticker only the chosen price type is left. 
        """
        # Check price_type argument. 
        available_price_type = list(self._orig_data.columns)
        if price_type not in available_price_type:
                raise ValueError("for price_type choose from: " + str(available_price_type))
        try:
            self.data = self._orig_data[price_type].copy()
            # self.data = self._orig_data[price_type].copy().reset_index(drop = True)
        except Exception: 
            if (price_type == "Adj Close") & (price_type not in available_price_type): 
                try: 
                    price_type = "Close"
                    print("Adj Close not available for this ticker, trying 'Close' instead!")
                    self.data = self._orig_data[price_type].copy()
                    # self.data = self._orig_data[price_type].copy().reset_index(drop = True)
                except:
                    raise ValueError("Choose one of the following price types: " + str(available_price_type)) 
                    
    def split_train_test(self, test_size):     
        """

        Parameters
        ----------
        test_size : float 0-1
            determines the size of the test set
            OR 
            a positive integer that's smaller thant the number of samples in self.data
            if test_size == 0, then the entire dataset is used for training. 

        Returns
        -------
        self.y_train + self.y_test
        """
        if test_size == 0: 
            self.y_train = self.data.copy()
            self.y_test = pd.DataFrame()
        elif test_size >= 1: 
            print()
            print("=========================")
            print("test_size >= 1, meaning that it stands for a number of samples not the relative size!")
            print("=========================")
            print()            
            self.y_train, self.y_test = temporal_train_test_split(self.data, 
                                        test_size= test_size)
        else: 
            self.y_train, self.y_test = temporal_train_test_split(self.data, 
                                        test_size= test_size)
       
        
#%% 
stock = DataImport()
stock.import_data(ticker = "AAPL", 
                  start = "2020-01-01", 
                  end = "2021-07-07", 
                        interval = "1d", group_by = "1d")
stock.prep_data("Close")
stock.split_train_test(test_size = 0)    

    