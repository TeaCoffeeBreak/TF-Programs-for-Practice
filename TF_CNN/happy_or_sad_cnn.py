
import tensorflow as tf
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
DESIRED_ACCURACY = 0.999
labels={0:'Happy',1:'Sad'}


class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["acc"]>0.999:
            print("ACcuracy reached 100%")
            self.model.stop_training=True
callback=MyCallBack()
def generator(path):
    train_gen= ImageDataGenerator(rescale=1/255.0)
    train_data=train_gen.flow_from_directory(directory=path, target_size=(300,300),class_mode="binary", batch_size=8)
    return train_data
def build_model():
    model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,3, activation="relu",input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
    return model
def plot_loss_acc(history):
    epochs= range(len(history.history['acc']))
    acc=history.history['acc']
    loss=history.history['loss']
    plt.plot(epochs,acc)
    plt.title("Accuracy")
    plt.figure()
    plt.plot(epochs,loss)
    plt.title("Loss")
    plt.show()

data_gen=generator("Datasets/h-or-s")
# show_predictions(None,data_gen)
model=build_model()
history=model.fit(data_gen, epochs=10, callbacks=[callback])
plot_loss_acc(history)