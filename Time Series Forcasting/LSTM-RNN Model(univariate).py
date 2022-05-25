import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler

# def building_dataset(data_series, data_window):
#
#     x = []
#     y = []
#
#     starting_index = 0
#     for last_index in range(data_window, len(data_series)):
#         if len(data_series[starting_index:last_index]) == data_window:
#             x.append(data_series[starting_index:last_index])
#             try:
#                 y.append(data_series[last_index])
#                 starting_index = starting_index + 1
#             except Exception as e:
#                 print("Handled Error:", e)
#                 break
#         else:
#             break
#
#     return np.array(x) , np.array(y)

def building_dataset(data, input_window, output_window, feature, split_ratio):
    data_series = data[feature].values
    data_series = data_series.reshape((len(data_series), 1))

    data_series_train = data_series[0:int((1-split_ratio)*len(data_series))]
    data_series_test =  data_series[int((1 - split_ratio) * len(data_series)):]

    scaler = MinMaxScaler()
    scaled_data_series_train = scaler.fit_transform(data_series_train)

    x_train = []
    y_train = []

    for i in range(input_window, len(scaled_data_series_train)):
        if len(scaled_data_series_train[i-input_window: i]) == input_window:
            x_train.append(scaled_data_series_train[i-input_window: i])
        else:
            break
        if len(scaled_data_series_train[i:i+output_window]) == output_window:
            y_train.append(scaled_data_series_train[i:i+output_window])
        else:
            x_train = x_train[:-1]
            break

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    scaled_data_series_test = scaler.fit_transform(data_series_test)
    x_test = []
    y_test = []


    for i in range(input_window, len(scaled_data_series_test)):
        if len(scaled_data_series_test[i-input_window: i]) == input_window:
            x_test.append(scaled_data_series_test[i-input_window: i])
        else:
            break
        if len(scaled_data_series_test[i:i+output_window]) == output_window:
            y_test.append(scaled_data_series_test[i:i+output_window])
        else:
            x_test = x_test[:-1]
            break

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train , x_test, y_test, scaler

if __name__ == '__main__':

    data = pd.read_csv(r"C:\Users\703294213\Downloads\NBCC.NS (2).csv")

    input_window = 30
    output_window = 1
    feature = 'Close'
    split_ratio = 0.1

    # data_series = [2543, 2640, 2718, 2782, 2758, 2695, 2775, 2778, 2819, 2790, 2780, 2693, 2640, 2620, 2518, 2474,
    #                2449, 2399, 2426]

    data.dropna(subset='Close', axis=0, inplace=True)
    # data_series = list(data['Close'][0:2480])
    # max = max(data_series)
    # data_series = list(np.array(data_series) / max)
    x_train, y_train, x_test, y_test, scaler = building_dataset(data, input_window, output_window, feature, split_ratio)

    '''For LSTM we need input reshaped in 3 dimention: no_of_records X no_of_inputs X n_features
    This n_features is just an add to make it 3 dimentions'''
    # n_features = 1
    # x = x.reshape((x.shape[0], x.shape[1], n_features))

    ##Diffrent approach
    # data_window = 7
    # dataset = data.iloc[:,4:5].values
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # scaled_dataset = scaler.fit_transform(dataset)
    # x_train = []
    # y_train = []
    # for i in range(data_window, len(dataset)-10):
    #     x_train.append(scaled_dataset[i-data_window:i, 0])
    #     y_train.append(scaled_dataset[i], 0)
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    #
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print("x_train shape : ", x_train.shape)

    '''Building LSTM model'''
    # model = Sequential()
    # model.add(LSTM(units = 100, activation = "relu", return_sequences = True, input_shape = (data_window, n_features)))
    # model.add(LSTM(units = 100, activation = "relu"))
    # model.add(Dense(1))
    model = Sequential()

    model.add(LSTM(50, activation = "relu", return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(50, activation = "relu" , return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(50, activation = "relu", return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(50, activation = "relu"))
    model.add(Dropout(0.25))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs = 300, verbose = 1)

    #Prediction
    y_pred = model.predict(x_test)
    y_test_2d = y_test.reshape((len(y_test),1))

    length_xtest = len(x_test)
    num_of_next_pred = 5  #m
    input_sequence = list(x_test[length_xtest-2].reshape(input_window))
    output_sequence = []
    for i in range(0,num_of_next_pred):
        input = input_sequence[i:]
        input = np.array(input).reshape((1,input_window,1))
        prediction = model.predict(input)
        output_sequence.append(prediction[0][0])
        input_sequence.append(prediction[0][0])

    actual = list(x_test[length_xtest-1].reshape(input_window))[0:num_of_next_pred]
    forcast = output_sequence

    actual = np.array(actual).reshape((len(actual), 1))
    forcast = np.array(forcast).reshape((len(forcast), 1))

    #Plotting graph to analyse the y_test and y_predicted value
    plt.plot(actual, color = 'red', label = 'Actual price')
    plt.plot(forcast, color = 'blue', label = 'Predicted price')
    plt.title('LSTM price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()




    print("Done")

