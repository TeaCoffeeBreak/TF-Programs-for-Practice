# Bird Detector (InceptionV3)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
import time
import pathlib
import numpy as np
train_dir = "Datasets/Indian_Birds/Training"
valid_dir = "Datasets/Indian_Birds/Validation"
BATCH_SI = 128
labels = {0:"Bulbul",1:"Myna",2: "Tailor Bird"}


def generator():
    train_data = ImageDataGenerator(rescale=1./255)
    valid_data = ImageDataGenerator(rescale=1./255)

    train_generator = train_data.flow_from_directory(train_dir,
                                             target_size=(150,150),
                                             batch_size=BATCH_SI,
                                             class_mode="categorical")
    valid_generator = valid_data.flow_from_directory(valid_dir,
                                             target_size=(150,150),
                                             batch_size=BATCH_SI,
                                             class_mode="categorical")
    return train_generator,valid_generator


class Callback1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["acc"] > 0.99:
            print("Accuracy reached 99%")
            self.model.stop_training=True


def plot_img(train_gen):
    sample_img,_=next(train_gen)
    fig,axes=plt.subplots(2,5, figsize=(20,20))
    axes= axes.flatten()
    for img,ax in zip(sample_img,axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def pretrained_model():
    inception_model = InceptionV3(input_shape=(150,150,3),include_top=False,weights="imagenet")
    for layer in inception_model.layers:
        layer.trainable=False
    return inception_model


def display(img,actual_label,pred_label):
    plt.figure(figsize=(15,15))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
    plt.text(0.1,0.5, 'Actual: '+labels[int(actual_label)], size=15, color='red')
    plt.text(0.1, 10, 'Prediction: ' + labels[int(pred_label)], size=15, color='blue')

    plt.axis("off")
    plt.show()


def show_predictions(model,data_gen):
    # print(len(data_gen.next()))
    # print(data_gen.next()[1][0])
    image=data_gen.next()
    pred_label= model.predict(image[0][0].reshape((1,150,150,3)))
    label=image[1][0]
    print(tf.argmax(label))
    print(label)
    display(image[0][0],tf.argmax(label),tf.argmax(pred_label[0]))


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs.keys())
        show_predictions(model,train_gen)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


def plot_loss_acc(history):
    loss=history.history["loss"]
    val_loss=history.history["val_loss"]
    acc=history.history["acc"]
    val_acc=history.history["val_acc"]
    epochs= range(len(acc))
    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.figure()
    plt.plot(epochs,acc)
    plt.plot(epochs,val_acc)
    plt.show()


def save_model():
    model.save("model_data")

# images=["Datasets/Indian_Birds/Tailorbird.jpg","Datasets/Indian_Birds/Bulbul.jpg","/Datasets/Indian_Birds/Validation/Myna/1200262.jpg","Datasets/Indian_Birds/Validation/Bulbul/1200245.jpg"]


def predict_image(model,images):
    img=[]
    for i in images:
        im=tf.keras.preprocessing.image.load_img(path=i,target_size=(150,150), color_mode="rgb")
        im = tf.keras.preprocessing.image.img_to_array(
            im, data_format=None, dtype=None
        )
        img.append(im)
    # img=np.array(img)
    prediction1=np.argmax(model.predict(img[0].reshape((1,150,150,3))))
    prediction2=np.argmax(model.predict(img[1].reshape((1,150,150,3))))
    prediction3=np.argmax(model.predict(img[2].reshape((1,150,150,3))))
    prediction4=np.argmax(model.predict(img[3].reshape((1,150,150,3))))
    print(prediction1,prediction2,prediction3,prediction4)


def load_model(path):
    model= tf.keras.models.load_model(path)
    return model


def tflite_model(modelpath,savepath):

    converter=tf.lite.TFLiteConverter.from_saved_model(modelpath)
    converter.optimizations= [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflitemodel=converter.convert()
    tflitemodel_file= pathlib.Path(savepath+str(time.time())+"model.tflite")
    tflitemodel_file.write_bytes(tflitemodel)
    return tflitemodel

# tflite_model()


def tflite_interpreter(tflitemodel,images):
    img = []
    for i in images:
        im = tf.keras.preprocessing.image.load_img(path=i, target_size=(150, 150), color_mode="rgb")
        im = tf.keras.preprocessing.image.img_to_array(
            im, data_format=None, dtype=None
        )
        img.append(im)
    img=np.array(img)
    interptreter= tf.lite.Interpreter(model_content=tflitemodel)
    interptreter.allocate_tensors()
    input_details=interptreter.get_input_details()[0]['index']
    output_details=interptreter.get_output_details()[0]['index']
    tflite_results=[]
    print("Predicting")
    for i in img:
        interptreter.set_tensor(input_details,i)
        interptreter.invoke()

        tflite_results.append(interptreter.get_tensor(output_details))
    print(tflite_results)


if __name__ == '__main__':

    train_gen, valid_gen = generator()
    callback = Callback1()

    plot_img(train_gen)
    dispcall = DisplayCallback()

    pre_trained_model = pretrained_model()
    last_layer = pre_trained_model.get_layer("mixed7")
    last_output = last_layer.output
    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(pre_trained_model.input, x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    history = model.fit_generator(train_gen,
                                  epochs=15,
                                  steps_per_epoch=1840 // BATCH_SI,
                                  validation_data=valid_gen,
                                  validation_steps=184 // BATCH_SI,
                                  callbacks=[dispcall]
                                  )
    # plot_loss_acc(history)
    # save_model()
    # model = load_model(path)
    # print(model.summary())
    # predict_image(model,images)
