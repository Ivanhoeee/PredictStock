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

class StockPredictor():

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
            self.data = self._orig_data[price_type].copy().reset_index(drop = True)
        except: 
            if (price_type == "Adj Close") & (price_type not in available_price_type): 
                try: 
                    price_type = "Close"
                    print("Adj Close not available for this ticker, trying 'Close' instead!")
                    self.data = self._orig_data[price_type].copy().reset_index(drop = True)
                except:
                    raise ValueError("Choose one of the following price types: " + str(available_price_type)) 
                    
    def split_train_test(self, test_size):         
        self.y_train, self.y_test = temporal_train_test_split(self.data, 
                                    test_size= test_size)
        
    def forecast(self):
        """
        This function estimates a number of pre-defined models. 
        For each model a forecast is made for self.y_test number of steps_ahead.

        Returns
        -------
        self.predictions: dataframe
            for each model a column in the dataframe is made. 
        """
        # Create forecasthorizon object 
        fh = np.arange(1, len(self.y_test) + 1) 
        
        y_train = self.y_train.copy()
        y_pred = pd.DataFrame()
        # Naive forecast (base model)
        forecaster1 = NaiveForecaster(strategy="last")
        forecaster1.fit(y_train)
        y_pred_1 = forecaster1.predict(fh)
        y_pred_1.name = "Naive_last"
        y_pred_1.index = self.y_test.index
        y_pred = pd.concat([y_pred, y_pred_1], axis = "columns")
        # Exp smoothing
        forecaster2 = ExponentialSmoothing()
        forecaster2.fit(y_train)
        y_pred_2 = forecaster2.predict(fh)
        y_pred_2.name = "Exp_smooth"        
        y_pred_2.index = self.y_test.index
        y_pred = pd.concat([y_pred, y_pred_2], axis = "columns") 
        # AutoArima
        forecaster3 = AutoARIMA()
        forecaster3.fit(y_train)
        y_pred_3 = forecaster3.predict(fh)
        y_pred_3.name = "AutoArima"
        y_pred_3.index = self.y_test.index
        y_pred = pd.concat([y_pred, y_pred_3], axis = "columns")   
        # put predictions back into self
        self.y_pred = y_pred.copy()
        
    def plot_pred(self): 
        """
        This function plots the predictions + y_test + y_train
        """
        plt.plot(stock.y_test, label = "y_test")
        plt.plot(stock.y_train, label = "y_train")        
        for variable in stock.y_pred.columns: 
            plt.plot(stock.y_pred[variable], label = variable + " MAPE: " + 
                     str(round(self.accuracy[variable].values[0],3)))
        plt.legend(loc = 0)   
        plt.tight_layout()
        plt.show()
        
    def calc_accuracy(self): 
        """
        This function calculates the mape for every prediction made.
        """
        
        accuracy = pd.DataFrame(index = ["MAPE"], 
                                columns = list(self.y_pred.columns))
        for prediction in accuracy.columns: 
            accuracy[prediction] = mean_absolute_percentage_error(self.y_test, self.y_pred[prediction])
        self.accuracy = accuracy
        
    def calc_profit(self): 
        """
        This function calculates the profits for a given trading strategy. 
        Profits are calculated as follows: 
            1. if FC up AND current position = 0: 
                    BUY
            2. if FC up AND current position = 1: 
                    HOLD/ DO NOTHING
            3. if FC down and current position = 0: 
                    HOLD/ DO NOTHING
            4. if FC up and current position = 1: 
                    SELL
        
        Parameters
        ---------
        y_pred: dataframe
        y_actual: dataframe

        Returns
        -------
        df: dataframe
            Dataframe that contains a time-series with profits 
        """
        df_list = [] # we save all the dataframes with profits in this list
        model_names = list(self.y_pred.columns) # these are the dict keys
        y_pred = self.y_pred.copy()
        y_actual = self.data.copy()
        
        # Get actuals and predictions into 1 dataframe
        for col in model_names: 
            df = y_pred[[col]].merge(y_actual, left_index = True, 
                                   right_index = True, how = "left")
            # Timeshift actual 1 day
            df[y_actual.name + "_yesterday"] =  df[y_actual.name].shift(1)
            # Add new columns
            df[["Position", "Profit"]] = 0
            # Loop to determine position & profit
            for i in range(len(df)): 
                if np.isnan(df.loc[df.index[i], y_actual.name + "_yesterday"]) == True: 
                    df.loc[df.index[i], "Position"] = 0                    
                elif (df.loc[df.index[i], col] > df.loc[df.index[i], 
                    y_actual.name + "_yesterday"]) & (df.loc[df.index[i-1], "Position"] == 0): 
                    df.loc[df.index[i], "Position"] = 1
                elif (df.loc[df.index[i], col] > df.loc[df.index[i], 
                    y_actual.name + "_yesterday"]) & (df.loc[df.index[i-1], "Position"] == 1): 
                    df.loc[df.index[i], "Position"] = df.loc[df.index[i-1], "Position"]
                elif (df.loc[df.index[i], col] < df.loc[df.index[i], 
                    y_actual.name + "_yesterday"]) & (df.loc[df.index[i-1], "Position"] == 0): 
                    df.loc[df.index[i], "Position"] = df.loc[df.index[i-1], "Position"] 
                elif (df.loc[df.index[i], col] < df.loc[df.index[i], 
                    y_actual.name + "_yesterday"]) & (df.loc[df.index[i-1], "Position"] == 1): 
                    df.loc[df.index[i], "Position"] = 0 
                    
            for i in range(1,len(df)): 
                # if position changes from 0 to 1 (we are buying hence our profit is negative since we bought)
                if (df.loc[df.index[i], "Position"] != df.loc[df.index[i-1], "Position"]) & \
                    (df.loc[df.index[i], "Position"] == 1): 
                    df.loc[df.index[i], "Profit"] = -df.loc[df.index[i], y_actual.name + "_yesterday"] + \
                                                    df.loc[df.index[i-1], "Profit"]
                elif (df.loc[df.index[i], "Position"] != df.loc[df.index[i-1], "Position"]) & \
                    (df.loc[df.index[i], "Position"] == 0): 
                    df.loc[df.index[i], "Profit"] = df.loc[df.index[i], y_actual.name + "_yesterday"]  + \
                                                    df.loc[df.index[i-1], "Profit"]
                else: 
                    df.loc[df.index[i], "Profit"] = df.loc[df.index[i-1], "Profit"]\
                                    
            df_list.append(df.dropna(axis = "rows"))          
        return dict(zip(model_names, df_list))
    
    def plot_profits(self): 
        for key in self.profits.keys(): 
            plt.plot(self.profits[key]["Profit"], label = key + " Cum. Profits")
        plt.title("Cumulative profits")
        plt.tight_layout()
        plt.legend(loc = 0)
        plt.show()
            
            
                    
                
                    
                

            
        
        
    def save_output(self): 
        """
        
        This function saves output to csv. 

        Returns
        -------
        None.

        """
        
#%%
# Test yahoo finance
yf.download(tickers = "AAPL", start = "2021-01-01",
            end = "2021-01-02", interval="1d")        
stock = StockPredictor()
# stock.import_data(ticker = "AAPL", 
#                   start = "2020-01-01", 
#                   end = "2021-07-07", 
#                         interval = "1d", group_by = "1d")
stock._orig_data = pd.DataFrame(data = np.random.uniform(low=100, high=200, size=(100,)), 
                                                         index = range(100), 
                                                         columns = ["Close"])

stock._orig_data = pd.DataFrame(data = (np.arange(100,200) + 
                                np.random.uniform(low = -30, high = 30, size = (100,))), 
                                index = range(100), columns = ["Close"])

stock.prep_data("Close")
stock.split_train_test(test_size = 0.25)
stock.forecast()
stock.calc_accuracy()
stock.plot_pred()
stock.profits = stock.calc_profit()
stock.plot_profits()

for key in stock.profits.keys(): 
    print(key)
    print(stock.profits[key]["Profit"].head())
    plt.plot(stock.profits[key]["Profit"], label = key + " Cum. Profits")
plt.legend(loc = 0)
plt.title("Cumulative profits")
plt.tight_layout()
plt.show()


profits = stock.profits.copy()