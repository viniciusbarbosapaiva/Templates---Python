# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# Importing the training set
df_raw = pd.read_csv('Google_Stock_Price_Train.csv')
df = df_raw
df.columns
df = df_raw.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
time_new = df['Date'].iloc[0]
df['Date'] = df['Date'].apply(lambda time_new: time_new.date())
df.info()
df.Close = df.Close.str.replace(',', '')
df['Close'] = df['Close'].apply(pd.to_numeric)

plt.figure(figsize=(14, 5), dpi=100)
plt.plot(df['Date'], df['Close'], label='Google_Stock_Price_Train')
plt.vlines(datetime.date(2016,1, 1), 0, 1300, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Google stock price')
plt.legend()
plt.show()

def get_technical_indicators(feature):
    
    # Create 7 and 21 days Moving Average
    df['ma7'] = df[feature].rolling(window=7).mean()
    df['ma20'] = df[feature].rolling(window=20).mean()
    
    
    # Create MACD
    df['26ema'] = pd.DataFrame.ewm(df[feature], span=26).mean()
    df['12ema'] = pd.DataFrame.ewm(df[feature], span=12).mean()
    df['MACD'] = (df['12ema']-df['26ema'])
    
    # Create Bollinger Bands
    df['20 Day STD'] = df[feature].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (df['20 Day STD']*2)
    df['lower_band'] = df['ma20'] - (df['20 Day STD']*2)
    
    # Create Exponential moving average
    df['ema'] = df[feature].ewm(com=0.5).mean()
    
    # Create Momentum
    df['momentum'] = df[feature]-1
    
    return df

df_TI = get_technical_indicators('Close')

def plot_technical_indicators(dataset, last_days, feature):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = df.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset_train_days = df.iloc[-last_days:, :]
    x_ = range(3, dataset_train_days.shape[0])
    x_ =list(dataset_train_days.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset_train_days['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset_train_days[feature],label='Open Price', color='b')
    plt.plot(dataset_train_days['ma20'],label='MA 20', color='r',linestyle='--')
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
plot_technical_indicators(df_TI, 400, 'Close')

#Splitting Train and Test Set
df = df.set_index(['Date'], drop=True)
train = df.iloc[:1006, 0:1 ]
test = df.iloc[1006:, 0:1]
plt.figure(figsize=(10, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1, 1))
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)

for s in range(1,2):
    train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
    test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)
    
X_train = train_sc_df.dropna().drop('Y', axis=1)
y_train = train_sc_df.dropna().drop('X_1', axis=1)

X_test = test_sc_df.dropna().drop('Y', axis=1)
y_test = test_sc_df.dropna().drop('X_1', axis=1)

X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print('Train shape: ', X_train_lmse.shape)
print('Test shape: ', X_test_lmse.shape)

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
regressor.add(LSTM(units = 50, return_sequences = False, input_shape = (X_train_lmse.shape[1], 1),
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
lstm_model = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
'''

# Fitting the RNN to the Training set
lstm_model = regressor.fit(X_train_lmse, y_train, epochs=100, batch_size=64)
y_pred_test_lstm = regressor.predict(X_test_lmse)
y_train_pred_lstm = regressor.predict(X_train_lmse)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))


# Part 3 - Making the predictions and visualising the results
predicted_stock_price = regressor.predict(X_test_lmse)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.figure(figsize=(10, 6))
plt.plot(sc.inverse_transform(y_test), label='True')
plt.plot(predicted_stock_price, label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Close')
plt.legend()
plt.show();
