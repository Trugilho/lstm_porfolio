#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:12:22 2022

@author: juliandro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:32:52 2022

@author: juliandro
"""
from tensorflow.keras.layers import LSTM, Flatten, Dense,Reshape,Lambda,Input
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
import yfinance as yf
import pandas as pd

def prepair_data(window_x,window_y):
    tickers = ['VTI', 'AGG', 'DBC', 'VIXY']
    
    
    df_list = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, group_by="Ticker", start="2015-10-01", end="2021-10-1")
        data['ticker'] = ticker
        data = data[['Close','ticker']]
        df_list = pd.concat([df_list,data])
        
    df_pivot = pd.pivot_table(df_list, values='Close',index=df_list.index.values,
                        columns=['ticker'])
    
    
    close = df_pivot.copy()
    daily_return = ((close.shift(-1) - close)/close).shift(1)
    daily_return = daily_return.iloc[1:]
    df_pivot = df_pivot.iloc[1:]
    
    tickers = df_pivot.columns

    X = df_pivot.values.reshape(df_pivot.shape[0],1,-1)
    y = daily_return.values
    #
#t = tf.constant(df_pivot)
#t = tf.cast(t, tf.float32)
    X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
    y = rolling_array(y[window_x:],stepsize=1,window=window_y)
    X = np.moveaxis(X,-1,1)
    # X1 = np.moveaxis(X1,-1,1)
    y = np.swapaxes(y,1,2)

    return X,y,tickers

def rolling_array(a, stepsize=1, window=60):
    n = a.shape[0]
    return np.stack((a[i:i + window:stepsize] for i in range(0,n - window + 1)),axis=0)

#X,y,tickers = prepair_data(window_x=64,window_y=19)


import numpy as np

# setting the seed allows for reproducible results

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self,timesteps_input=64,timesteps_output=19):
        self.data = None
        self.model = None
        
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
        
        X,y,tickers = prepair_data(window_x=timesteps_input,window_y=timesteps_output)

        self.X_tr,self.X_val,self.y_tr,self.y_val = train_test_split(X, y, 
                                                 test_size=0.2,shuffle=False)

    def __build_model(self):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([
            LSTM(64,  input_shape=(4,64)),
            Flatten(),
            Dense(4, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def get_allocations(self):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
#        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        
#        data = data.iloc[1:]
#        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model()
        
        self.model.fit(self.X_tr, self.y_tr, epochs=50, shuffle=False)
        return self.model.predict(self.X_val)
    
model = Model()
allocations = model.get_allocations()
