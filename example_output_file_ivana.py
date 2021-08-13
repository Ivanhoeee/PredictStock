# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:38:11 2021

@author: P870306
"""

example = {"SP500": {
  "actuals": [
      {"date": "20-07-2021", "value": "100.0"},
      {"date": "21-07-2021", "value": "100.1"},
      {"date": "22-07-2021", "value": "100.0"}
    ],
  "forecast": {
    "NaiveForecaster": {
      "forecast":[
      {"date": "23-07-2021", "value": "100.5"},
      {"date": "24-07-2021", "value": "100.9"},
      {"date": "25-07-2021", "value": "112.5"}
      ],
      "MAPE": 0.04,
      "ROI": 0.07},

    "ExponentialSmoothing": {
      "forecast":[
      {"date": "23-07-2021", "value": "100.7"},
      {"date": "24-07-2021", "value": "100.6"},
      {"date": "25-07-2021", "value": "111.2"}
      ],
      "MAPE": 0.03,
      "ROI": 0.09},

    "AutoARIMA": {
      "forecast":[
      {"date": "23-07-2021", "value": "100.0"},
      {"date": "24-07-2021", "value": "100.2"},
      {"date": "25-07-2021", "value": "105.3"}
      ],
      "MAPE": 0.015,
      "ROI": 0.11}
  }
}
}