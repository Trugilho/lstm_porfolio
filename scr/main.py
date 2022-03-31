import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from lstm_model import build_lstm_model
from preprocessing import prepair_data
from utils import *
from sklearn.model_selection import train_test_split


# -*- coding: utf-8 -*-
class portfolio_optmization:
    def __init__(self,path_data,timesteps_input=64,timesteps_output=19):
        self.model_name = 'LSTM'
        X,y,tickers = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
        hyper_params = {
          "activation": "sigmoid",
          "l2": 0.04727381534237722,
          "l2_1": 0.07275854097919236,
          "l2_2": 0.07628361490840738,
          "units": 32
        }
        self.X,self.y,self.tickers = X,y,tickers
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
#        
        self.hyper_params = hyper_params
        self.hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])
        self.model = build_lstm_model(self.hyper_params)

    def train_model(self,n_fold,batch_size,epochs):
#        tscv = TimeSeriesSplit(n_splits=n_fold)
#        for train_index, test_index in tscv.split(self.X):
#    
#            X_tr, X_val = self.X[train_index], self.X[test_index[range(self.timesteps_output-1,len(test_index),self.timesteps_output)]]
#            y_tr, y_val = self.y[train_index], self.y[test_index[range(self.timesteps_output-1,len(test_index),self.timesteps_output)]]
#            his = self.model.fit(X_tr, y_tr, batch_size=batch_size, epochs= epochs,validation_data=(X_val,y_val))
#            mask_tickers = self.predict_portfolio(X_val)
#            print('Sharpe ratio of this portfolio: %s' % str([self.calc_sharpe_ratio(mask_tickers[i],y_val[i]) for i in range(len(y_val))]))
        X_tr,X_val,y_tr,y_val = train_test_split(self.X, self.y, 
                                                 test_size=0.2,shuffle=False)
        his = self.model.fit(X_tr, y_tr, batch_size=batch_size, epochs= epochs,validation_data=(X_val,y_val))
        mask_tickers = self.predict_portfolio(X_val)
        print('Sharpe ratio of this portfolio: %s' % str([self.calc_sharpe_ratio(mask_tickers[i],y_val[i]) for i in range(len(y_val))]))

    def predict_portfolio(self,X):
        results = self.model.predict(X)
        mask_tickers = results>0.5
        print("There are total %d samples to predict" % len(results))
        for i in range(len(mask_tickers)):
            for j in range(len(self.tickers)):
                if mask_tickers[i][j] == 1:
                    print('Sample %d -> Ticker: %s - Weight: %s' % (i, self.tickers[j],str(results[i][j]) ))
#            print('Sample %d : [ %s ]' % (i, ' '.join([self.tickers[j] for j in range(len(self.tickers)) if mask_tickers[i][j]==1])))
    
        return mask_tickers
    
    
    def calc_sharpe_ratio(self,weight,y):
        """Here y is the daily return have the shape (tickers,days)
        weight have the shape (tickers,)"""
        epsilon = 1e-6
        weights = np.round(weight)
        sum_w = np.clip(weights.sum(),epsilon,y.shape[0])
        norm_weight = weights/sum_w
        port_return = norm_weight.dot(y).squeeze()
        mean = np.mean(port_return)
        std = np.maximum(np.std(port_return),epsilon)
        return np.sqrt(self.timesteps_output) * mean/std

data_path = '/home/juliandro/Documentos/Mestrado/Deep Learning/portfolio_optimization/data/data.csv'
delafo = portfolio_optmization(data_path)
delafo.train_model(n_fold=10,batch_size=16,epochs=50)