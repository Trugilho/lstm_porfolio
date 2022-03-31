#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:14:38 2022

@author: juliandro
"""

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
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from keras.models import load_model,save_model

def get_data(start_date='2014-01-01', end_date='2018-01-01'):
    #def prepair_data(window_x,window_y):
    tickers = ['VTI', 'AGG', 'DBC', 'VIXY']
    
    
    df_list = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, group_by="Ticker", start=start_date, end=end_date)
        data['ticker'] = ticker
        data = data[['Close','ticker']]
        df_list = pd.concat([df_list,data])
        
    df_pivot = pd.pivot_table(df_list, values='Close',index=df_list.index.values,
                        columns=['ticker'])
    return df_pivot
 
def get_return_data(data):
    assets = data.columns
    portfolio_return_data = np.concatenate([data.pct_change().values[1:] ], axis=1)
    portfolio_return_data_df = pd.DataFrame(portfolio_return_data,  columns =assets,index=data.index.values[1:],)  
    return portfolio_return_data_df


    
class Model:
    def __init__(self,timesteps_input=64,timesteps_output=19):
        self.data = None
        self.model = None
        
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
        
#        X,y,tickers = prepair_data(window_x=timesteps_input,window_y=timesteps_output)
#
#        self.X_tr,self.X_val,self.y_tr,self.y_val = train_test_split(X, y, 
#                                                 test_size=0.2,shuffle=False)
#        
#        self.input_shape = (X.shape[1],X.shape[2])
        
    def build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  

            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = (K.mean(portfolio_returns) * 255) / (K.std(portfolio_returns)* np.sqrt(255))
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
    

        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def train(self,data):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        tickers = data.columns
        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.build_model((None,8), len(data.columns))
        
        fit_predict_data = data_w_ret[np.newaxis,:]        
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=50, shuffle=False)
        return self.model
def get_allocations(model,data,allocation_date=None):
    
    # data with returns
    data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
    fit_predict_data = data_w_ret[np.newaxis,:]        
    tickers = data.columns
    df = pd.DataFrame(model.predict(fit_predict_data)[0])
    if allocation_date is not  None:
        print("Optiomal Allocation for date %s" % allocation_date)
    list_allocations = {}
    for i in range(0,len(tickers)):
        list_allocations[tickers[i]] = df[0][i]
        print(tickers[i],df[0][i])
    
    return list_allocations

def portfolio_return(portfolio_prices_data):

    portfolio_prices_data = portfolio_prices_data
#    portfolio_prices_data.index = portfolio_prices_data.index.strftime('%Y-%m-%d')

    trained_model = load_model('my_model.h5',compile=False)
    
    allocations = get_allocations(trained_model,portfolio_prices_data)
    assets = portfolio_prices_data.columns
    portfolio_return_data = np.concatenate([portfolio_prices_data.pct_change().values[1:] ], axis=1)
    portfolio_return_data_df = pd.DataFrame(portfolio_return_data,  columns =assets,index=portfolio_prices_data.index.values[1:])  
#    allocations_df = pd.DataFrame(allocations, index=[0])
    portfolio_daily_return_data_df = pd.DataFrame()
    for asset in assets:
        portfolio_daily_return_data_df[asset + ' return'] = portfolio_return_data_df[asset]*allocations[asset]
    portfolio_daily_return_data_df['daily_return'] = portfolio_daily_return_data_df.sum(axis = 1)
    portfolio_daily_return_data_df['cumulative_ret'] = (portfolio_daily_return_data_df['daily_return'] + 1).cumprod()
    return portfolio_daily_return_data_df


#
#model = Model()
#train_data = get_data()
###
#model_trained = model.train(train_data)
#save_model = model_trained.save('my_model.h5')

#
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

def mv_portfolio(portfolio_prices_data):
    mu = mean_historical_return(portfolio_prices_data)
    S = CovarianceShrinkage(portfolio_prices_data).ledoit_wolf()
    
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    #print(cleaned_weights)
    ef.portfolio_performance(verbose=True)
    assets = portfolio_prices_data.columns.to_list()
    teste = portfolio_prices_data.pct_change()[1:]
    teste_annual = teste.mean()*252
    teste_annual_pivot = pd.DataFrame(teste_annual)
    teste_annual_pivot = teste_annual_pivot.pivot_table(columns=teste_annual_pivot.index.values)
    for asset in assets:
        teste_annual_pivot[asset] = teste_annual_pivot[asset] * cleaned_weights[asset]
    teste_annual_pivot['anual'] = teste_annual_pivot.sum(axis = 1)
    teste_data_with_return_mv = portfolio_prices_data.pct_change()[1:]
    
    for asset in assets:
        teste_data_with_return_mv[asset] = teste_data_with_return_mv[asset] * cleaned_weights[asset]
        
    teste_data_with_return_mv['daily_return'] = teste_data_with_return_mv.sum(axis = 1)
    teste_data_with_return_mv['cumulative_ret'] = (teste_data_with_return_mv['daily_return'] + 1).cumprod()
    return teste_data_with_return_mv

#
#
#fig = plt.figure()
#ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
#ax1.plot(port_return['cumulative_ret'] )
#
#ax1.set_xlabel('Date')
#ax1.set_ylabel("Cumulative Returns")
#ax1.set_title("Portfolio Cumulative Returns")
#
#ax2 = ax1.twinx()
#ax1.plot(port_return['cumulative_ret'] )
#
#plt.show();
#

