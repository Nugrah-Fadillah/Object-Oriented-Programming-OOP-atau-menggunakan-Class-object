# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:30:51 2024

@author: ASUS
"""

import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Class StockPredictor
class StockPredictor:
    def __init__(self, stock_symbol, period='5y'):
        self.stock_symbol = stock_symbol
        self.period = period
        self.stock_data = None
        self.model = LinearRegression()
        
    #Downloading data from Yahoo Finance
    def download_data(self):
        self.stock_data = yf.download(self.stock_symbol, period=self.period)
        
        #Change data to ordinal
        self.stock_data['Date'] = pd.to_datetime(self.stock_data.index). map(pd.Timestamp.toordinal)
    
    #Making linier regression
    def train_model(self):
        X = self.stock_data['Date'].values.reshape(-1,1)
        y = self.stock_data['Close'].values
        self.model.fit(X,y)
        
    #Predicting future stock price
    def predict_future(self, days_ahead=365):
        X = self.stock_data['Date'].values.reshape(-1,1)
        
        #Predicting amount of days in the future
        future_dates = np.array([X[-1] + i for i in range(days_ahead)]).reshape(-1,1)
        predicted_prices = self.model.predict(future_dates)
        
        #Changing ordinal date to datetime format
        future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(date[0])) for date in future_dates])
        return future_dates, predicted_prices
        
    #Plotting prediction result
    def plot_results(self, future_dates, predicted_prices):
        X = self.stock_data['Date'].values.reshape(-1,1)
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Harga Aktual')
        plt.plot(self.stock_data.index, self.model.predict(X), label='Trend (Interpolasi)', linestyle='--')
        plt.plot(future_dates, predicted_prices, label='Prediksi (1 tahun ke depan)', linestyle=':')
        
        plt.legend()
        plt.title(f'Prediksi Harga Saham {self.stock_symbol}')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga')
        plt.show()
        
        
#Class application
predictor = StockPredictor('NVO', period='5y')
predictor.download_data()           #step 1: dowloading stock data
predictor.train_model()             #step 2: training linier regression model
future_dates, predicted_prices = predictor.predict_future(365)          #step 3: predicting future price
predictor.plot_results(future_dates, predicted_prices)                   #step 4:showing result on plot
