#Bird Detector (3 layer CNN )
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
train_dir="Datasets/Indian_Birds/Training"
valid_dir="Datasets/Indian_Birds/Validation"
BATCH_SI=128
def generator(train_dir,valid_dir):
    train_data=ImageDataGenerator(rescale=1./255)
    valid_data=ImageDataGenerator(rescale=1./255)

    train_gen= train_data.flow_from_directory(train_dir,
                                              target_size=(150,150),
                                              batch_size=BATCH_SI,
                                              class_mode="categorical")
    valid_gen=  valid_data.flow_from_directory(valid_dir,
                                               target_size=(150,150),
                                               batch_size=BATCH_SI,
                                               class_mode="categorical")
    return  train_gen,valid_gen
train_gen,valid_gen=generator(train_dir,valid_dir)
def plot_img(train_gen):
    sample_img,_= next(train_gen)
    fig,axes=plt.subplots(1,5,figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(sample_img,axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
plot_img(train_gen)
class Callback1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["acc"]>0.99:
            print("Accuracy reached 99%")
            self.model.stop_training=True
callback= Callback1()
def build_model(train_gen,valid_gen):
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),input_shape=(150,150,3),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(3,activation="softmax")
    ])
    print(model.summary())
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['acc'])
    return model
model=build_model(train_gen,valid_gen)
history=model.fit_generator(train_gen,
                    steps_per_epoch=1203//BATCH_SI,
                    epochs=15,
                    validation_data=valid_gen,
                    validation_steps=1,
                    callbacks=[callback]
                            )
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
plot(history)
def save_model():
    model.save("model_data")
save_model()
images=["Datasets/Indian_Birds/Tailorbird.jpg","Datasets/Indian_Birds/Bulbul.jpg"]
def predict_image():
    img=[]
    for i in images:
        img.append(tf.keras.preprocessing.image.load_img(path=i,target_size=(150,150)))
    img=np.array(img)
    prediction=model.predict(img)
    print(prediction)
predict_image()