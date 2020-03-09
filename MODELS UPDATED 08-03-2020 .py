#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting with Python (SARIMA, LSTM, BPNN, Neural ODEs.)

# ## Group: Minter

# 
# ### Table of Contents
# - <a href="#step1">1: Library/Package and data Loading</a>
# - <a href="#step2">2: SARIMA</a>
# - <a href="#step3">3: LSTM</a>
# - <a href="#step4">4: BPNN</a>
# - <a href="#step5">5: NEURAL ODES</a>
# - <a href="#step6">6: Results</a>
# 

# <h3>1. Library and data loading</h3>
# <a id="step1"></a>

# In[1]:


import numpy as np
import pandas as pd
import os
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 

#from pmdarima import auto_arima 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#from random import seed


# In[2]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # FORECAST

# ## Read Dataset

# In[3]:


#Test dataset
df = pd.read_csv('../beer.csv')


# In[4]:


#Outputs the first five rows of the datasaet
df.head()


# In[5]:


df.info()


# In[6]:


#Chahging the month col to datetime from object/string
df.Month = pd.to_datetime(df.Month)


# In[7]:


#sets the index of  the dataframe to month col
df = df.set_index("Month")
df.head()


# In[8]:


#Sets the frequency as montly
df.index.freq = 'MS'


# In[9]:


#Plots the original dataset
plt.figure(figsize=(18,9))
plt.plot(df.index, df["Monthly beer production"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Production')
plt.show();


# ## SARIMA Forecast
# <a id="step2"></a>

# In[10]:


#Firstly, we take a closer look at the dataset
a = seasonal_decompose(df["Monthly beer production"], model = "add")
a.plot();


# In[11]:


#We run auto_arima() function to get best p,d,q,P,D,Q values
#Then, we use sarimax to account for seasonality and then forecasting
#building the model

from pmdarima.arima import auto_arima

model = auto_arima(df['Monthly beer production'],trace=True, error_action='ignore', suppress_warnings=True, seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4).summary()
#model.summary()


# In[12]:


#Prints the model results
model


# As we can see best arima model chosen by auto_arima() is SARIMAX(2, 1, 3)x(3, 0, [1], 12)
# 

# In[13]:


#Let's split the data into train and test set
train_data = df[:len(df)-12]
test_data = df[len(df)-12:]

#plotting the data
ax = train_data.plot(figsize=(15,10))
test_data.plot(ax=ax,figsize=(15,10))


# In[14]:


#Building the SARIMAX Model to account for seasonality
sarima_model = SARIMAX(train_data['Monthly beer production'], order = (2,1,3), seasonal_order = (3,0,1,12))
sarima_model = sarima_model.fit()
sarima_model.summary()


# In[15]:


#As the model was built, fitting and trained, now we can predict the results
sarima_pred = sarima_model.predict(start = len(train_data), end = len(df)-1, typ="levels").rename("SARIMA_Predictions")

#Prints the results
sarima_pred


# In[16]:


test_data['SARIMA_Predictions'] = sarima_pred


# In[17]:


#Compares the oirginal dataset to the predicted data using the SARIMA model
test_data


# In[18]:


#Ploting the predicted data aginst the original dataset for the test data
test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)
sarima_pred.plot(legend = True,linestyle="--");


# In[19]:


#Errors which would be used to esitmate a model accuarcy

sarima_rmse_error = rmse(test_data['Monthly beer production'],test_data['SARIMA_Predictions'])
#sarima_rmse_error = mean_squared_error(test_data['Monthly beer production'],test_data['SARIMA_Predictions'], squared=False)

sarima_mse_error = sarima_rmse_error**2
#sarima_mse_error = mean_squared_error(test_data['Monthly beer production'],test_data['SARIMA_Predictions'])

sarima_mae_error = mean_absolute_error(test_data['Monthly beer production'],test_data['SARIMA_Predictions'])
sarima_mape_error = mean_absolute_percentage_error(test_data['Monthly beer production'],test_data['SARIMA_Predictions'])

mean_value = df['Monthly beer production'].mean()

print(f'MSE Error: {sarima_mse_error}\nRMSE Error: {sarima_rmse_error}\nMAE: {sarima_mae_error}\nMAPE: {sarima_mape_error}\nMean: {mean_value}')


# ## LSTM Forecast
# <a id="step3"></a>

# In[20]:


#loading libraies for LSTM and BPNN 
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

#First we'll scale our train and test data with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[21]:


#we scale the data to normalise the figues in order to utilize them as we cannot we them as natural numbers
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[22]:


#Before creating LSTM model we should create a Time Series Generator object.

n_input = 12
n_features= 1
generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)


# In[23]:


#building the LSTM model
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(tf.keras.layers.Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

#printing the model resulst
lstm_model.summary()


# In[24]:


#Fitting the model using the generator previouslt created
lstm_model.fit_generator(generator,epochs=20)


# In[25]:


#Plotting the losses from each epoch iteration
losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);


# In[26]:


#Batch training the model 
lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[27]:


#Printing the results of the training loop
lstm_predictions_scaled


# In[28]:


#As you know we scaled our data that's why we have to inverse it to see true predictions.
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)


# In[29]:


#prints inverse results
lstm_predictions


# In[30]:


test_data['LSTM_Predictions'] = lstm_predictions


# In[31]:


#Comparison of models  
test_data


# In[32]:


#Plotting LSTM model againts the original test data
test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)
test_data['LSTM_Predictions'].plot(legend = True,linestyle="--");


# In[33]:


#Errors which would be used to esitmate a model accuarcy

lstm_rmse_error = rmse(test_data['Monthly beer production'], test_data["LSTM_Predictions"])
#lstm_rmse_error = mean_squared_error(test_data['Monthly beer production'],test_data["LSTM_Predictions"], squared= False)

lstm_mse_error = lstm_rmse_error**2
#lstm_mse_error = mean_squared_error(test_data['Monthly beer production'],test_data["LSTM_Predictions"])

lstm_mae_error = mean_absolute_error(test_data['Monthly beer production'],test_data["LSTM_Predictions"])
lstm_mape_error = mean_absolute_percentage_error(test_data['Monthly beer production'],test_data["LSTM_Predictions"])

mean_value = df['Monthly beer production'].mean()

print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMAE: {lstm_mae_error}\nMAPE: {lstm_mape_error}\nMean: {mean_value}')


# ## BPNN Forecast
# <a id="step4"></a>

# In[34]:


#As we previously loaded the libraies we required and sacled the dataset already
#We begin by building the NN model, Uisng 1 input layer, 1 hidden layer and 1 output layer 

bp_model = tf.keras.Sequential()
bp_model.add(tf.keras.layers.Dense(200,activation='relu', input_shape=(n_input, n_features)))
bp_model.add(tf.keras.layers.Dense(100, activation='relu'))
bp_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
bp_model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

#prints the model results
bp_model.summary()


# In[35]:


#Fitting the model using the generator previously created
bp_model.fit_generator(generator,epochs=20)


# In[36]:


#Plotting the losses across epoch cycles
losses_bp = bp_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_bp)),losses_bp);


# In[37]:


bp_predictions_scaled = list()


# In[38]:


#Batch training the model
batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    bp_pred = bp_model.predict(current_batch)[0][i]
    bp_predictions_scaled.append(bp_pred) 


# In[39]:


#Training results
bp_predictions_scaled


# In[40]:


#As we scaled the data, we now have to inverse it to see true predictions.
bp_predictions = scaler.inverse_transform(bp_predictions_scaled)


# In[41]:


#Prediction results
bp_predictions


# In[42]:


test_data['BP_Predictions'] = bp_predictions


# In[43]:


#Plotting the BPNN model againts the original test data
test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)
test_data['BP_Predictions'].plot(legend = True,linestyle="--");


# In[44]:


#Errors which would be used to esitmate a model accuarcy

bp_rmse_error = rmse(test_data['Monthly beer production'], test_data["BP_Predictions"])
#bp_rmse_error = mean_squared_error(test_data['Monthly beer production'],test_data["BP_Predictions"], squared= False)

bp_mse_error = bp_rmse_error**2
#bp_mse_error = mean_squared_error(test_data['Monthly beer production'],test_data["BP_Predictions"])

bp_mae_error = mean_absolute_error(test_data['Monthly beer production'],test_data["BP_Predictions"])
bp_mape_error = mean_absolute_percentage_error(test_data['Monthly beer production'],test_data["BP_Predictions"])

mean_value = df['Monthly beer production'].mean()

print(f'MSE Error: {bp_mse_error}\nRMSE Error: {bp_rmse_error}\nMAE: {bp_mae_error}\nMAPE: {bp_mape_error}\nMean: {mean_value}')


# ## Neural ODEs Forecast
# <a id="step5"></a>

# In[ ]:





# ## RESULTS
# <a id="step6"></a>

# In[45]:


#####---Final Prediction of all models

#Plotting all models against each other
plt.figure(figsize=(16,8))
plt.plot_date(test_data.index, test_data["Monthly beer production"], label="Monthly beer production",linestyle="-")
plt.plot_date(test_data.index, test_data["SARIMA_Predictions"], label="SARIMA Predictions",linestyle="-.")
plt.plot_date(test_data.index, test_data["LSTM_Predictions"], label="LSTM Predictions",linestyle="--")
plt.plot_date(test_data.index, test_data["BP_Predictions"], label="BP Predictions",linestyle=":")
plt.legend()
plt.show()


# In[52]:


###########----Errors
#Creating a list of all errors previously calcuated
rmse_errors = [sarima_rmse_error, lstm_rmse_error, bp_rmse_error]
mse_errors = [sarima_mse_error, lstm_mse_error, bp_mse_error]
mae_errors = [sarima_mae_error, lstm_mae_error, bp_mae_error]
mape_errors = [sarima_mape_error, lstm_mape_error, bp_mape_error]

#creating a new df to store the errors accordingly 
errors = pd.DataFrame({"Models" : ["SARIMA", "LSTM", "BP"],"RMSE Errors" : rmse_errors, "MSE Errors" : mse_errors, "MAE Errors" : mae_errors, "MAPE Errors" : mape_errors})

#printing the mean of the original dataset
print(f"Mean: {test_data['Monthly beer production'].mean()}")

#setting the inde of the new df to Models
errors = errors.set_index("Models")

#printing the results of all models for comparison
errors


# In[53]:


#Lastly, we print the predictions from all models for the period of the test data against the original 
test_data


# In[ ]:





# In[ ]:




