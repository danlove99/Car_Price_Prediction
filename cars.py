import pandas as pd
import numpy as np

df = pd.read_csv('USA_cars_datasets.csv')
y = df['price'].values
df = df.drop('price', axis=1)
df = df.drop('vin', axis=1)
df = pd.get_dummies(df, dummy_na=True)
print(df.values.shape)
X = df.values
X = X[:, 1:]
X = (X-X.min())/(X.max()-X.min())*20
print(X[0])

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(308, input_dim=308, activation='relu'))
model.add(Dense(616, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(308, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
model.fit(X,y, epochs=10, batch_size=100)