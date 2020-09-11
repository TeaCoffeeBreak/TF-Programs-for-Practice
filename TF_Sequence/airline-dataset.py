import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
def get_data():
    df=pd.read_csv("Datasets/Airline/monthly-airline-passengers.csv")
    return df.Month.values, df.Passengers.values
def preprocess(x):

    x=scaler.fit_transform(x.reshape([-1,1]))
    return x

def windowed_dataset(series, windowsi, batch_si, shuffle_buffer):
    dataset=tf.data.Dataset.from_tensor_slices(series)
    dataset=dataset.window(windowsi+1,shift=1,drop_remainder=True)
    dataset=dataset.flat_map(lambda window:window.batch(windowsi+1))
    dataset=dataset.shuffle(shuffle_buffer).map(lambda window:(window[:-1],window[-1]))
    dataset=dataset.batch(batch_si).prefetch(1)
    return dataset
months,data=get_data()
data=preprocess(data)
print(len(data))
split_time=104
time=range(len(data))
x_train=data[:split_time]
print(len(x_train))
x_test=data[split_time:]
time_train=time[:split_time]
time_valid=time[split_time:]
# data=tf.data.Dataset.range(144)
batchsi=4
windowsi=5
dataset=windowed_dataset(x_train,windowsi,batchsi,int(len(data)*0.3))

# for x,y in dataset:
#     print("x=",len(x.numpy()))
#     print("y=",len(y.numpy()))

def build_model(dataset,batchsi, windowsi):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Dense(10,input_shape=[windowsi],activation="relu"),
        tf.keras.layers.Dense(10,activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return model
def build_model_rnn(dataset,batchsi, windowsi):
    def op(X):
        return X*100
    model=tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]),
        tf.keras.layers.LSTM(40,return_sequences=True),
        tf.keras.layers.LSTM(40),
        tf.keras.layers.Dense(1),
        # tf.keras.layers.Lambda(lambda X:op(X))
    ])
    return  model
scheduler=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 *10**(epoch/20))
model=build_model_rnn(dataset,batchsi,windowsi)
model.compile(loss="mse",optimizer="adam",metrics="mae")
# history=model.fit(dataset,epochs=100, callbacks=[scheduler])
history=model.fit(dataset,epochs=400)
def plot_loss(history):
    plt.semilogx(history.history['lr'],history.history["loss"])
    plt.axis([1e-8,1e-3, 0,0.1   ])
    plt.show()
# plot_loss(history)
def plot_loss_mae(history):
    epoch=range(len(history.history["loss"]))
    loss=history.history["loss"]
    mae=history.history["mae"]
    plt.plot(epoch,loss)
    plt.plot(epoch,mae)
    plt.show()
plot_loss_mae(history)