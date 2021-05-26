# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
dataset_train['Date'] = pd.to_datetime(dataset_train['Date'], infer_datetime_format=True)
time_new = dataset_train['Date'].iloc[0]
dataset_train['Date'] = dataset_train['Date'].apply(lambda time_new: time_new.date())

plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset_train['Date'], dataset_train['Open'], label='Google_Stock_Price_Train')
plt.vlines(datetime.date(2016,4, 20), 0, 800, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Google stock price')
plt.legend()
plt.show()

def get_technical_indicators(feature):
    
    # Create 7 and 21 days Moving Average
    dataset_train['ma7'] = dataset_train[feature].rolling(window=7).mean()
    dataset_train['ma21'] = dataset_train[feature].rolling(window=21).mean()
    
    
    # Create MACD
    dataset_train['26ema'] = pd.DataFrame.ewm(dataset_train[feature], span=26).mean()
    dataset_train['12ema'] = pd.DataFrame.ewm(dataset_train[feature], span=12).mean()
    dataset_train['MACD'] = (dataset_train['12ema']-dataset_train['26ema'])
    
    # Create Bollinger Bands
    dataset_train['21 Day STD'] = dataset_train[feature].rolling(window=20).std()
    dataset_train['upper_band'] = dataset_train['ma21'] + (dataset_train['21 Day STD']*2)
    dataset_train['lower_band'] = dataset_train['ma21'] - (dataset_train['21 Day STD']*2)
    
    # Create Exponential moving average
    dataset_train['ema'] = dataset_train[feature].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset_train['momentum'] = dataset_train[feature]-1
    
    return dataset_train

dataset_train_TI = get_technical_indicators('Open')

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset_train.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset_train_days = dataset_train.iloc[-last_days:, :]
    x_ = range(3, dataset_train_days.shape[0])
    x_ =list(dataset_train_days.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset_train_days['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset_train_days['Open'],label='Open Price', color='b')
    plt.plot(dataset_train_days['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset_train_days['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset_train_days['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset_train_days['lower_band'], dataset_train_days['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Google - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset_train_days['MACD'],label='MACD', linestyle='-.')
    plt.hlines(0, xmacd_, shape_0, colors='g', linestyles='--')
    plt.legend()
    plt.show()
plot_technical_indicators(dataset_train_TI, 400)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
days = 20
for i in range(days, len(dataset_train)): #60 por causa de 60 dias ou 03 meses. 1258 é o tamanho do banco de dados estudado.
    X_train.append(training_set_scaled[i-days:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
reshaping_train = X_train.shape[1]
n = 1 # Número de indicadores (Volume, preço de fechamento, etc.)
X_train = np.reshape(X_train, (X_train.shape[0], reshaping_train , n))
print('Train shape: ', X_train.shape)
print('Test shape: ', y_train.shape)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False, input_shape = (reshaping_train, n),
                   activation='relu', kernel_initializer='lecun_uniform')) # units = 50 é um bom número
#return_sequences = False se tiver apenas um layer. Se houver mais de um layer marcar True.
regressor.add(Dropout(0.2)) # Dropout = 0.2 é um bom número

'''
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, activation='relu'))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, activation='relu'))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
'''
# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# optimizer = adam é um bom optimizer.  
# loss = Como RNN é um problema de regressão, 'mean_squared_error'

# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
'''
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
history = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
'''

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs=100, batch_size=64)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - days:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(days, len(real_stock_price)+days):
    X_test.append(inputs[i-days:i, 0])
X_test = np.array(X_test)
reshaping_test = X_test.shape[1]
X_test = np.reshape(X_test, (X_test.shape[0], reshaping_test, n))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
