import  tensorflow as tf
import csv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "/home/dj/Downloads/sign_language/sign_mnist_train/sign_mnist_train.csv"
test_path = "/home/dj/Downloads/sign_language/sign_mnist_test/sign_mnist_test.csv"
def get_data(path):

    images,labels=[],[]
    with open(file=train_path,mode="r") as train_file:
        reader=csv.reader(train_file)
        next(reader)
        for row in reader:

            label=row[0]
            img=row[1:]
            img=np.array(img).astype(dtype="float64").reshape((28,28))

            images.append(img)
            label=float(label)
            labels.append(label)
    images=np.array(images)

    labels=np.array(labels)
    return images,labels
training_images,training_labels= get_data(train_path)
print(training_images.shape,training_labels.shape)
testing_images,testing_labels=get_data(test_path)
print(training_labels)
training_images=np.expand_dims(training_images,axis=-1)
testing_images=np.expand_dims(testing_images,axis=-1)
print(testing_images.shape,testing_labels.shape)
train_data= ImageDataGenerator(rescale=1/255.)
test_data=ImageDataGenerator(rescale=1/255.)

train_gen=train_data.flow(training_images,training_labels,batch_size=32)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,3, activation="relu",input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(25,"softmax")
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["acc"])
model.fit_generator(train_gen, steps_per_epoch=len(training_images)//32, epochs=10)

