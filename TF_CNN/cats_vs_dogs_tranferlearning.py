#Transfer Learning Cats vs Dogs
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
import numpy as np
import matplotlib.pyplot as plt
import os
def get_data():
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir=os.path.join(PATH,"train")
    valid_dir= os.path.join(PATH,'validation')
    train_cat=os.path.join(train_dir,'cats')
    train_dog=os.path.join(train_dir,'dogs')
    valid_cat=os.path.join(valid_dir,'cats')
    valid_dog=os.path.join(valid_dir,'dogs')

    print("Number of training cat images",len(os.listdir(train_cat)))
    print("Number of training dog images" , len(os.listdir(train_dog)))
    print("Number of validation cat images" , len(os.listdir(valid_cat)))
    print("Number of validation cat images" , len(os.listdir(valid_dog)))
    total_train= len(os.listdir(train_cat)) +len(os.listdir(train_dog))
    total_valid= len(os.listdir(valid_cat))+len(os.listdir(valid_dog))
    return train_dir,valid_dir,total_train,total_valid
train_dir,valid_dir,total_train,total_valid=get_data()
BATCH_SI=128
def generator():
    train_data= ImageDataGenerator(rescale=1./255)
    valid_data=ImageDataGenerator(rescale=1./255)
    train_gen= train_data.flow_from_directory(train_dir,
                                              target_size=(150,150),
                                              batch_size=BATCH_SI,
                                              class_mode="binary")
    valid_gen = valid_data.flow_from_directory(valid_dir,
                                               target_size=(150,150),
                                               batch_size=BATCH_SI,
                                               class_mode="binary")
    return train_gen,valid_gen
train_gen,valid_gen = generator()
def plot_img():
    sample_img,_=next(train_gen)
    fig, axes= plt.subplots(1,5, figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(sample_img,axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
plot_img()

def pretrained_model():
    pre_trained_model= InceptionV3(input_shape=(150,150,3), include_top=False,weights='imagenet')
    for layer in pre_trained_model.layers:
        layer.trainable=False
    pre_trained_model.summary()
    last_layer= pre_trained_model.get_layer("mixed7")
    last_output=last_layer.output
    return pre_trained_model,last_output

pre_trained_model,last_output=pretrained_model()
x=tf.keras.layers.Flatten()(last_output)
x= tf.keras.layers.Dense(512,activation="relu")(x)
x=tf.keras.layers.Dense(1,activation="sigmoid")(x)
model=tf.keras.Model(pre_trained_model.input,x)
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
history=model.fit_generator(train_gen,
                    steps_per_epoch= total_train//BATCH_SI,
                    epochs=15,
                    validation_data=valid_gen,
                    validation_steps=total_valid//BATCH_SI
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