import tensorflow as tf

mnist= tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["acc"]>=0.99:
            print("Accuracy Reached 99%.Training will be stopped")
            self.model.stop_training=True
callback=MyCallback()
model= tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    # tf.keras.layers.Dense(32,activation="relu"),
    # tf.keras.layers.Dense(64,activation="relu"),
    # tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(512,activation="relu"),
    # tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(x_train,y_train,epochs=10,callbacks=[callback])
