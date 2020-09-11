#Binary Image Classifiacrion Cats vs Dogs
import tensorflow as tf
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import os
import matplotlib.pyplot as plt
from Models.ResNet50 import ResNet50
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
def generator(train_dir,valid_dir):
    # train_data= ImageDataGenerator(rescale=1./255)
    # valid_data=ImageDataGenerator(rescale=1./255)
    #augmentation
    train_data=ImageDataGenerator(rescale=1/255.,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode="nearest")
    valid_data= ImageDataGenerator(rescale=1/255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest"
    )
    train_gen= train_data.flow_from_directory(train_dir,
                                              target_size=(150,150),
                                              batch_size=128,
                                              class_mode="binary")
    valid_gen= valid_data.flow_from_directory(valid_dir,
                                              target_size=(150,150),
                                              batch_size=128,
                                              class_mode="binary")
    return train_gen, valid_gen

train_gen,valid_gen= generator(train_dir,valid_dir)
def plot_img(train_gen):
    sample_train_img,_ = next(train_gen)
    fig,axes =plt.subplots(1,5, figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(sample_train_img,axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
plot_img(train_gen)
def build_model():
    # model=tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(64,(3,3),padding='same',input_shape=(150,150,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     # tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    #     # tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     # tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512,activation='relu'),
    #     tf.keras.layers.Dense(1,activation='sigmoid')
    # ])
    resnet= ResNet50()
    model = resnet.resnet50(input_shape=(150,150,3), classes=2)
    print(model.summary())
    return model

class CallBack1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["acc"]>0.95:
            print("Accuracy reached 95")
            self.model.stop_training=True
callback= CallBack1()
model= build_model()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
history=model.fit_generator(train_gen,
                    epochs=15,
                    validation_data=valid_gen,
                    steps_per_epoch=total_train//128,
                    validation_steps=total_valid//128,
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