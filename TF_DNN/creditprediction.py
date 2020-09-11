#Neural network for binary classification credit score
import tensorflow as tf
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
def preprocessing(path="Datasets/CreditCard/creditcard.csv"):
    df = pd.read_csv(path)
    print(df.head(6))
    df=df.values
    np.random.shuffle(df)
    x=df[:,:-1]
    mean= np.mean(x)
    std=np.std(x)
    x_norm= (x-mean)/std
    y=df[:,-1]
    y = y.reshape([-1,1])
    print(x[:4,:])
    return x_norm,y
x,y=preprocessing()
print(x.shape, y.shape)
def train_valid_test_split(x,y):

    x_train=x[:int(len(x)*0.8),:]
    y_train=y[:int(len(y)*0.8),:]
    x_valid=x[int(len(x)*0.8):,:]
    y_valid=y[int(len(y)*0.8):,:]
    x_test=x_valid[int(len(x_valid)*0.5):,:]
    y_test=y_valid[int(len(y_valid)*0.5):,:]
    x_valid=x_valid[:int(len(x_valid)*0.5),:]
    y_valid=y_valid[:int(len(y_valid)*0.5),:]
    return  x_train,x_valid,x_test,y_train,y_valid,y_test
x_train,x_valid,x_test, y_train,y_valid,y_test=train_valid_test_split(x,y)
print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape,y_test.shape)

def build_model():
    model= tf.keras.Sequential([
        tf.keras.layers.Dense(units=30,input_shape=x_train.shape,activation='relu'),
        tf.keras.layers.Dense(units=128,activation='relu'),
        tf.keras.layers.Dense(units=1, activation="sigmoid")
    ])
    return model
class CallBack1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss']<0.5:
            print("Accuracy Reached 95")
            self.model.stop_training=True
def train():
    callback=CallBack1()
    model=  build_model()
    print(model.summary())
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    history=model.fit(x=x_train,y=y_train,epochs=10,validation_data=(x_valid,y_valid), callbacks=[callback], batch_size=64, validation_batch_size=64)
    return history,model
history,model= train()
print("Evaluating accuracy")
print(model.evaluate(x_test, y_test))

def plot(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss= history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.title("Loss")
    plt.figure()
    plt.plot(epochs,acc)
    plt.plot(epochs,val_acc)
    plt.title("Accuracy")
    plt.show()