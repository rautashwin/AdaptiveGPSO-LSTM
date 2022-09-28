

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import itertools
import random
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[140]:


get_ipython().system('pip install ann_visualizer')


# In[141]:


from ann.visualizer.visualize import ann_viz


# In[2]:


data = pd.read_csv('Combine.csv', parse_dates=True,index_col='TIMESTAMP')


# <a id="2"></a> <br>
# ## Descriptive Analysis

# In[3]:


data.head()


# In[4]:


print(data.columns)
print(data.shape)


# In[5]:


data=data.drop(['extID'], axis=1)


# In[6]:


data = data.drop(['status', 'avgMeasuredTime','medianMeasuredTime',
     'vehicleCount', '_id', 'REPORT_ID'],axis=1)


# In[7]:


data


# In[ ]:





# In[8]:


data=data.iloc[:32080,:]


# In[9]:


data['avgSpeed1']=data['avgSpeed'].ewm(span=2).mean()


# In[10]:


data


# In[11]:


data=data.iloc[:32080,1:3]


# In[12]:


data


# In[13]:


data_1=data.iloc[:,:]


# In[14]:


data_1


# <a id="24"></a> <br>
# ## Fearure Scaling

# In[15]:


#data_1.to_csv('data_with_timestamp.csv')


# In[16]:


# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
data_scaled= sc.fit_transform(data)


# In[17]:


tf.keras.utils.plot_model


# In[18]:


data_scaled.shape


# In[19]:


# Creating a data structure (it does not work when you have only one feature)
def create_data(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []
    
    for i in range(n_past, len(df) - n_future + 1):
        x_data.append(df[i - n_past:i, 0:n_feature])
        y_data.append(df[i + n_future - 1:i + n_future, 0])
    
    split_training_test_starting_point = int(round(train_test_split_percentage*len(x_data)))
    split_train_validation_starting_point = int(round(split_training_test_starting_point*(1-validation_split_percentage)))
    
    x_train = x_data[:split_train_validation_starting_point]
    y_train = y_data[:split_train_validation_starting_point]
    
    # if you want to choose the validation set by yourself, uncomment the below code.
    x_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]
    y_val =  x_data[split_train_validation_starting_point:split_training_test_starting_point]                                             
    
    x_test = x_data[split_training_test_starting_point:]
    y_test = y_data[split_training_test_starting_point:]
    
    return np.array(x_train), np.array(x_test), np.array(x_val), np.array(y_train), np.array(y_test), np.array(y_val)


# In[20]:


# Number of days you want to predict into the future
# Number of past days you want to use to predict the future

X_train, X_test, X_val, y_train, y_test, y_val = create_data(data_scaled, n_future=1, n_past=25, train_test_split_percentage=0.8,
                                               validation_split_percentage = 0)


# In[21]:


print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


# In[22]:


y_test


# <a id="3"></a> <br>
# ## Train LSTM Model

# In[23]:


# ------------------LSTM-----------------------
'''''
regressor = Sequential()
regressor.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=256, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='adam', loss='mse')
#regressor.fit(X_train, y_train, epochs=100, batch_size=64)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# fit model
history = regressor.fit(X_train, y_train, validation_split=0.3, epochs=1000, batch_size=64, callbacks=[es])
'''''



# In[24]:


# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


def LSTM_HyperParameter_Tuning(config, x_train, y_train, x_test, y_test):
    
    first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = config
    possible_combinations = list(itertools.product(first_additional_layer, second_additional_layer, third_additional_layer,
                                                  n_neurons, n_batch_size, dropout))
    
    print(possible_combinations)
    print('\n')
    print(print(np.asarray(possible_combinations).shape))
    hist = []
    
    for i in range(0, len(possible_combinations)):
        
        print(f'{i}th combination: \n')
        print('--------------------------------------------------------------------')
        
        first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = possible_combinations[i]
        
        # instantiating the model in the strategy scope creates the model on the TPU
        #with tpu_strategy.scope():
        regressor = Sequential()
        regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        regressor.add(Dropout(dropout))

        if first_additional_layer:
            regressor.add(LSTM(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        if second_additional_layer:
            regressor.add(LSTM(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        if third_additional_layer:
            regressor.add(GRU(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        regressor.add(LSTM(units=n_neurons, return_sequences=False))
        regressor.add(Dropout(dropout))
        regressor.add(Dense(units=1, activation='linear'))
        regressor.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        '''''
        From the mentioned article above --> If a validation dataset is specified to the fit() function via the validation_data or v
        alidation_split arguments,then the loss on the validation dataset will be made available via the name “val_loss.”
        '''''

        file_path = 'best_model.h5'

        mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        '''''
        cb = Callback(...)  # First, callbacks must be instantiated.
        cb_list = [cb, ...]  # Then, one or more callbacks that you intend to use must be added to a Python list.
        model.fit(..., callbacks=cb_list)  # Finally, the list of callbacks is provided to the callback argument when fitting the model.
        '''''

        regressor.fit(x_train, y_train, validation_split=0.3, epochs=1000, batch_size=n_batch_size, callbacks=[es, mc], verbose=0)

        # load the best model
        # regressor = load_model('best_model.h5')

        train_accuracy = regressor.evaluate(x_train, y_train, verbose=0)
        test_accuracy = regressor.evaluate(x_test, y_test, verbose=0)

        hist.append(list((first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout,
                          train_accuracy, test_accuracy)))

        print(f'{str(i)}-th combination = {possible_combinations[i]} \n train accuracy: {train_accuracy} and test accuracy: {test_accuracy}')
        
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        
    return hist


# In[25]:


config = [[False,True], [False,True], [False,True], [128,256], [64,32], [0.1,0.2]]  

# list of lists --> [[first_additional_layer], [second_additional_layer], [third_additional_layer], [n_neurons], [n_batch_size], [dropout]]

hist = LSTM_HyperParameter_Tuning(config, X_train, y_train, X_test, y_test)  # change x_train shape


# In[21]:


35


# <a id="44"></a> <br>
# ## Choosing the Best Model

# In[129]:


hist = pd.DataFrame(hist)
hist = hist.sort_values(by=[7], ascending=True)
hist


# In[130]:


hist.to_csv('Arhus_loss.csv')


# In[134]:


regressor.summary()


# In[146]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


# In[147]:


plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[152]:


regressor.layers


# <a id="5"></a> <br>
# ## Results

# In[90]:


print(f'Best Combination: \n first_additional_layer = {hist.iloc[0, 0]}\n second_additional_layer = {hist.iloc[0, 1]}\n third_additional_layer = {hist.iloc[0, 2]}\n n_neurons = {hist.iloc[0, 3]}\n n_batch_size = {hist.iloc[0, 4]}\n dropout = {hist.iloc[0, 5]}')


# In[91]:


first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = list(hist.iloc[0, :-2])


# In[92]:


regressor = Sequential()
regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(dropout))

if first_additional_layer:
    regressor.add(LSTM(units=n_neurons, return_sequences=True))
    regressor.add(Dropout(dropout))

if second_additional_layer:
    regressor.add(LSTM(units=n_neurons, return_sequences=True))
    regressor.add(Dropout(dropout))

if third_additional_layer:
    regressor.add(GRU(units=n_neurons, return_sequences=True))
    regressor.add(Dropout(dropout))

regressor.add(LSTM(units=n_neurons, return_sequences=False))
regressor.add(Dropout(dropout))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

file_path = 'best_model.h5'

mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

regressor.fit(X_train, y_train, validation_split=0.3, epochs=100, batch_size=n_batch_size, callbacks=[es, mc], verbose=0)


# In[93]:


regressor.evaluate(X_test, y_test)


# In[100]:


y_test_data = pd.DataFrame(y_test, columns = ['Column_A'])


# In[101]:


y_test_data.to_csv('psoin.csv')


# In[102]:


y_pred_data = pd.DataFrame(y_pred, columns = ['Column_A']) 


# In[104]:


y_pred_data.to_csv('psoout.csv')


# In[94]:


y_test.to_csv('psoin.csv')


# In[95]:


y_pred.to_csv('psoout.csv')


# In[38]:


y_test_=sc.inverse_transform(y_test)


# In[39]:


y_pred_=sc.inverse_transform(y_pred)


# In[40]:


y_test_


# In[41]:


y_pred_


# In[27]:


y_pred = regressor.predict(X_test)

plt.figure(figsize=(16,8), dpi= 300, facecolor='w', edgecolor='k')

plt.plot(sc.inverse_transform(y_test), color='red', label = 'Real Opening Price')
plt.plot(sc.inverse_transform(y_pred), color='green', label = 'Predicted Opening Price')
plt.legend(loc='best')


# In[32]:


from sklearn.metrics import r2_score


# In[33]:


# Define a function to calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    rsq=r2_score(actual, predictions)

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('R Square Error: {:.4f}'.format(rsq))
    print('')


# In[34]:


evaluate_prediction(sc.inverse_transform(y_pred), sc.inverse_transform(y_test), 'PSOLSTM')


# In[28]:


return_rmse(X_test,y_pred)


# As it is clear in the plot, the trend (rise and fall) of the stock price is well predicted. Nice!

# # If you liked my work then please upvote, Thank you.
