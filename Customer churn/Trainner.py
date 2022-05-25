import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = pd.read_csv(r"C:\Users\703294213\Documents\Work\Project Work\My Projects\Customer curn\Customer_churn_dataset.csv")
    print(data.head(5))

    '''Removing customerID feature as it wont add any value to our model'''
    data.drop(labels = ['customerID'], axis = 1, inplace = True)

    '''Handling any NULL or blank data'''
    #data.isnull().any()   #Checking null for each features
    #data.isnull().any().any()  #Checking if entire dataset has any null values

    # Getting information on empty values in dataset and removed the records of such case
    empty = {}
    for col in data.columns:
        col_list = []
        for i in range(len(data[col])):
            if data[col][i] == ' ':
                col_list.append(i)
            else:
                pass
        empty[col] = col_list

    for key,value in empty.items():
        if len(value) == 0:
            pass
        else:
            data.drop(value, axis=0, inplace=True)

    #Converting totalCharges into numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

    num = []
    cat_binary = []
    cat_multi = []

    for col in data.columns:
        if col == 'Churn':
            pass
        else:
            if data[col].dtypes == 'O':
                if len(data[col].unique()) == 2:
                    cat_binary.append(col)
                else:
                    cat_multi.append(col)
            elif data[col].dtypes == 'int64':
                num.append(col)
            elif data[col].dtypes == 'float64':
                num.append(col)


    print("Numerical columns: ", num)
    print("Categorical(Binary): ", cat_binary)
    print("Categorical(Multi): ", cat_multi)

    #Assigning 0/1 to binary categorical columns
    # for col in cat_binary:
    #     for i in range(len(data)):
    #         if data[col].iloc[i] == data[col].unique()[0]:
    #             data[col].iloc[i] = 0
    #         else:
    #             data[col].iloc[i] = 1

    #Creating dummy variable for multi category collumns & Binary
    data = pd.get_dummies(data, columns=cat_multi)
    data = pd.get_dummies(data, columns=cat_binary)

    #Scalling the numerical columns
    scaler = MinMaxScaler()
    data[num] = scaler.fit_transform(data[num])

    data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    #Train test split
    X = data.drop('Churn', axis='columns')
    Y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    #Building a Model
    input_layers = []
    output_layers = []

    inputs = keras.Input(shape=(45,))
    dense = layers.Dense(100, activation='relu')(inputs)
    dense = layers.Dropout(0.3)(dense)
    dense = layers.Dense(70, activation = 'relu')(dense)
    dense = layers.Dropout(0.2)(dense)
    dense = layers.Dense(30, activation='relu')(dense)
    dense = layers.Dropout(0.1)(dense)
    outputs = layers.Dense(1, activation='sigmoid')(dense)

    input_layers.append(inputs)
    output_layers.append(outputs)

    model = keras.Model(inputs=input_layers, outputs=output_layers, name="churn_model")
    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(optimizer = "adam",
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    # hidden layer=1, acc=82%, 2 | hidden_layer=2, acc=83% | hidden=3, with more neurons, acc=93%

    model.fit(X_train, y_train, epochs = 100, verbose = 2)

    print("done")

