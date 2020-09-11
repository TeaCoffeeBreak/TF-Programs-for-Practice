# Tensorflow CNN Programs
Convolutional Neural Networks are mostly used for image processing tasks such as object classification, object detection, image captioning, object localisation etc.
CNN's use filters to extract features from image. Tensorflow has Convolutional layer in its keras api.
##### Syntax of conv layer for 2d image: 
tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation, kernel_initializer, ...)

#### General steps to follow while solving CNN based problems:
 1. Split image dataset in training and validation(you can use TFs ImageDataGenerator)
 2. Resize images in dataset (both training and validation).
 3. Normalize images in dataset.
 4. If dataset is small then you can use augmentation. (First build model without augmentation then you can try it for increasing acc)
 5. Build a cnn model. (First try to build a simple model with less layers then try to tune it)
 6. Compile and fit model on the training data.(use callbacks for saving time)
 7. Validate on validation dataset.                                                         
 (Don't use transfer learning directly on simple datasets. It can waste your time and size of model created will be larger)
 
 
 #### Problem statements for the program
 1. MNIST: mnist is classification dataset for handwritten digits from 0-9. This dataset is available in tensorflow. It is a very popular dataset and you can start your cnn journey with this dataset.
 2. Sign mnist: This sign language dataset consist of 27455 training and 7172 test images data of 25 different hand signs. This dataset is available in csv format.
 3. Happy or Sad: This is binary emotion classification dataset which consist of 40 images of each happy and sad emoji. 
 4. Cats and Dogs: This dataset is subset of original cats-v-dogs dataset with 1500 images of cats and dogs. 
 5. Indian Birds dataset: This is custom dataset I have created for practice. It consist of 1840 training and 184 validation images of three types of birds Bulbul, Myna and Tailor bird 