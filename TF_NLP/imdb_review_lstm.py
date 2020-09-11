import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_token="<oov>"
num_epochs=10
tokeniser= Tokenizer(num_words=vocab_size, oov_token=oov_token)
word_index=None
def convert_to_tokens(sentences):
    global word_index
    sequences=tokeniser.texts_to_sequences(sentences)
    padded_sequences=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

    return padded_sequences
def get_data():

    imdb,info= tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train,test= imdb["train"],imdb["test"]
    training_data=[]
    training_labels=[]
    for s,l in train:
        training_data.append(str(s.numpy()))
        training_labels.append(l.numpy())
    testing_data=[]
    testing_labels=[]

    for s,l in test:
        testing_data.append(str(s.numpy()))
        testing_labels.append(l.numpy())
    training_labels=np.array(training_labels)
    testing_labels=np.array(testing_labels)
    tokeniser.fit_on_texts(training_data)
    training_data=convert_to_tokens(training_data)
    testing_data=convert_to_tokens(testing_data)

    return training_data,training_labels,testing_data,testing_labels

training_data,training_labels,testing_data,testing_labels=get_data()
word_index=tokeniser.word_index
def build_model(training_data,training_labels,testing_data,testing_labels):
    model=tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(6,activation="relu"),
            tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
    print(model.summary())
    history=model.fit(training_data,training_labels,
              epochs=num_epochs,
              validation_data=(testing_data,testing_labels))
    return history,model
history,model=build_model(training_data,training_labels,testing_data,testing_labels)
def plot(history):
    acc=history.history["acc"]
    val_acc=history.history["val_acc"]
    loss=history.history["loss"]
    val_loss=history.history["val_loss"]
    epochs=range(len(acc))
    plt.plot(epochs,acc)
    plt.plot(epochs,val_acc)
    plt.title("Accuracy")
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title("Loss")
    plt.show()
plot(history)

reverse_index=dict([(value,key) for (key,value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_index.get(i,'?') for i in text ])
