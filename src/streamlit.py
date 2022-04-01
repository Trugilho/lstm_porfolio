import streamlit as st
import pandas as pd
import numpy as np
from  main import get_return_data,get_allocations,get_data,portfolio_return,mv_portfolio
import datetime as dt
import matplotlib.pyplot as plt
from keras.models import load_model


sns.set()
st.title('Portfolio Optimization')
st.subheader('US Total Stock Index (VTI)')
st.subheader('US Aggregate Bond Index (AGG)')
st.subheader('US Commodity Index (DBC)')
st.subheader('Volatility Index (VIX)')
start = st.date_input(
     "Start Date",
     dt.date(2018, 1, 1))

end = st.date_input(
     "End Date",
     dt.date(2019, 10, 1))

start_date = start
end_date = end

data_load_state = st.text('Loading data...')
data = get_data(start_date=start_date, end_date=end_date)
data_load_state.text("Done! (using st.cache)")

#if st.checkbox('Show raw data'):
st.subheader('Prices')
st.write(data)

st.subheader('Assets Returns')
return_data = get_return_data(data)
st.write(return_data)

for asset in return_data.columns:
    
    fig, ax = plt.subplots()
    ax.set_title(asset + " Histogram Returns")
    ax.hist(return_data[asset], bins=20)
    st.pyplot(fig)

#
trained_model = load_model('my_model.h5',compile=False)
allocations = get_allocations(trained_model,data)
allocations_df = df = pd.DataFrame(allocations, index=[0])
st.subheader('Portfolio Allocation')

st.write(allocations_df)



port_return = portfolio_return(data)
mv = mv_portfolio(data)

st.subheader('Portfolio Cumulative Returns')

st.write(port_return)

fig, ax1 = plt.subplots()
#ax1 = fig.add_axes([0.5,0.5,1.5,1.5])
ax1.plot(port_return['cumulative_ret'] ,label = "LSTM Model")
ax1.plot(mv['cumulative_ret'] ,label = "Mean Variance Allocation")
plt.legend(loc="upper left")

ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("Portfolio Cumulative Returns")
plt.gcf().autofmt_xdate()
st.pyplot(fig)
