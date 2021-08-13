import numpy as np
import pandas as pd
import sktime
import yfinance as yf
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter

import numpy as np
import pandas as pd
# from sktime.forecasting.compose import ReducedRegressionForecaster
import sktime.forecasting.compose as compose
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR

class Model(): 
    """
    This class is responsible for estimating a model. 
    """
    
    def __init__(self, data_import, model_name): 
        self.model_name = model_name
        self.y_train = data_import.y_train.copy()
        self._y_train_index = self.y_train.index
        self.y_test = data_import.y_test.copy()
        self._y_test_index = self.y_test.index
        
    def create_fc_horizon(self): 
        if len(self.y_test) == 0: 
            self.fh = np.arange(1, 2) 
        else: 
            self.fh = np.arange(1, len(self.y_test) + 1)
            
    def fit_model(self): 
        if self.model_name == "NaiveForecaster": 
            self.model = self.Naive()
        elif self.model_name == "AutoARIMA": 
            self.model = self.AutoARIMA()
        elif self.model_name == "AutoETS": 
            self.model = self.AutoETS()
        else: 
            raise Exception("Please choose an implemented model!")
            
    def Naive(self, start_train = None, end_train = None, start_fc = None, 
              end_fc = None): 
        if start_train != None: 
            y_train = self.y_train.copy()
            y_train = y_train[(y_train.index >= start_train) & (y_train.index <= end_train)].copy()
            self.y_train = y_train.copy()
        return NaiveForecaster(strategy="last").fit(self.y_train.reset_index(drop = True))  

    def AutoARIMA(self): 
        return AutoARIMA().fit(self.y_train.reset_index(drop = True))
    
    def AutoETS(self): 
        return AutoETS(auto = True).fit(self.y_train.reset_index(drop = True))
    
    def forecast_out_of_sample(self, initial_window):
        """
        This function creates for y_train an out of sample forecast. 
        Let say y_train has len = 100. 
        initial_window = 70
        This means that the first model will be fitted on the first 70 datapoints.
        Then a forecast will be made for 71st point. 
        Then the model will be refitted with the 71st point and a forecast
        will be made for 72nd point etc. 
        The model returns the 30 out-of-sample 1 period ahead forecasts. 

        Parameters
        ----------
        initial_window : integer
            number of periods of y_train used to train the initial model.
            The difference in len(y_train) and initial_window is the number of 
            periods for which an out-of-sample fc will be made.

        Returns
        -------
        y_pred_out_of_sample: pandas series
            Series containing the out_of_sample fc. 

        """
        # Save index for the prediction window
        index_predictions = self.y_train.index[(initial_window+1):]
        y_train = self.y_train.copy().reset_index(drop = True) # function cannot handle datetime index yet

        # fit, predict, update 
        cv = ExpandingWindowSplitter(step_length=1, fh=[1], 
                                     initial_window= initial_window)
        pred_out_of_sample = evaluate(forecaster= self.model, y=y_train, cv=cv, 
                        strategy="refit", return_data=True)
        # Get predictions out of pred_out_of_sample object
        pred = []
        for i in range(len(df)): 
            pred.append(pred_out_of_sample["y_pred"][i].values[0])
        # transform pred into series with index_predictions
        self.y_pred_out_of_sample = pd.DataFrame(pred, index = index_predictions, 
                                    columns = [y_train.name + " Forecast"])
   
    # def out_of_sample_forecast(self, out_of_sample_size): 
    #     """
    #     This function creates one day ahead out-of-sample forecasts given y_train. 
    #     All the dates need to be within y_train. 
    #     Say y_train starts at 01-01-2020 and ends at 01-01-2021. 
    #     if the user takes the following parameters: 
    #         out_of_sample_forecast(start_train = 01-01-2020, end_train = 01-12-2020, 
    #                                start_fc = 02-12-2020, end_fc = 01-01-2021), 
    #         the model will be firstly fit 
    #     It fits the model given start_train and end_train. 
    #     Then it creates 
        
    #     Parameters
    #     ----------
    #     y_train: dataframe 
        
    #     out_of_sample_size: integer 1 or greater than 1
    #                 test_size stands for the number of periods that we want 
    #                 to create the out-of-sample forecast for. 
        
    #     start_train: string
        
    #     end_train: string
        
    #     start_fc: string
        
    #     end_fc: string

    #     Returns
    #     -------
    #     None.

    #     """
    #     # create initial y_train 
    #     y_train = self.y_train.copy()
    #     y_train_temp = y_train.loc[(y_train.index >= start_train) & 
    #                                (y_train.index <= end_train)].copy()
        
    #     y_train_temp, y_test_temp = temporal_train_test_split(y_train, 
    #                                     test_size = out_of_sample_size)
        
    #     y_pred = pd.DataFrame(index = y_test_temp.index, columns = y_test_temp.columns + " Forecast")
        
    #     fh= np.arange(1, 2) 
        
    #     mdl = self.model()
    #     # Loop
    #     for i in range(len(y_pred)): 
    #         if i == 0:
    #             y_pred.iloc[i] = mdl.predict(fh = fh)
    #         else: 
    #             # Update model
    #             mdl.update(y_test_temp.iloc[i-1])
    #             # Fit model
    #             mdl.fit(y_train_temp)
    #             # predict 
    #             y_pred.iloc[i] = mdl.predict(fh = fh)
                
        
    def forecast(self): 
        self.y_pred = self.model.predict(self.fh)
        
    def set_fc_index(self, index): 
        """
        This function sets the index of the forecast. If we forecast out-of-sample
        then we don't know the exact dates due to weekend and holidays.
        
        index: datetime index
        """
        self.y_pred.index = [index] 
        
    def mape(self): 
        """
        This function returns the mape given y_test and prediction. 
        """
        return mean_absolute_percentage_error(self.y_test, self.y_pred)
        
    def export_fc(self): 
        """ 
        This function exports the forecast in the JSON format.
        """
        export = {}
        export[self.model_name] = {}
        export[self.model_name]["forecast"] = []
        for i in range(len(self.y_pred)):
            day_fc = {}
            day_fc["date"] = self.y_pred.index[i].strftime("%Y-%m-%d")
            day_fc["value"] = str(self.y_pred.iloc[0])
            export[self.model_name]["forecast"].append(day_fc)
        return export

#%% 
model = Model(stock, "NaiveForecaster")
model = Model(stock, "AutoETS")
model.create_fc_horizon()
model.fit_model()
model.forecast()
model.set_fc_index(pd.to_datetime("2021-07-07"))
model.forecast_out_of_sample(initial_window = 375)
export = model.export_fc()
        


