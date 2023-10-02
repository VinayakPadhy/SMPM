#!/usr/bin/env python
# coding: utf-8

# In[27]:


### Importing Librarires


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


### Importing Dataset


# In[21]:


stock_data = yf.download('AAPL', start = '2016-01-01', end = '2023-01-01')
stock_data.head()


# In[ ]:


### Visualisign Stock Price History 


# In[22]:


plt.figure(figsize=(15,8))
plt.title('Stock Price History')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Price')


# In[ ]:


### Cleaning Dataset


# In[23]:


close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))


# In[ ]:


### Data Preprocessing


# In[25]:


train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 


# In[26]:


test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[28]:


### LSTM Setup


# In[33]:


# model = keras.Sequential()
# model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(layers.LSTM(100, return_sequences=True))
# model.add(layers.LSTM(50, return_sequences=False))
# model.add(layers.Dense(25))
# model.add(layers.Dense(1))
# model.summary()


# In[41]:


### For BiDirectional 


# In[47]:


from keras.layers import Bidirectional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional


# In[52]:


model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(x_train.shape[1], 1)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1, activation="ReLU"))
model.compile(loss = 'mean_squared_error', optimizer = 'Adam')


# In[35]:


### Training LSTM Model


# In[53]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 64, epochs=50)


# In[54]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[57]:


data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(15,8))
plt.title('Stock Price Prediction Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
plt.show()


# In[ ]:




