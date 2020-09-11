import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
vocab_si=10000
embedding_dim=16
trunc_type="post"
oov_token="<OOV>"
max_length=120
num_epochs=10
tokeniser=Tokenizer(num_words=vocab_si,oov_token=oov_token)
def convert_padded_tokens(sentences):
    sequences=tokeniser.texts_to_sequences(sentences)
    padded_sequences=pad_sequences(sequences=sequences,maxlen=max_length,truncating=trunc_type)
    return padded_sequences
def get_data():
    imdb,info=tfds.load("imdb_reviews",with_info=True,as_supervised=True)
    train,test= imdb["train"],imdb["test"]
    training_data=[]
    training_labels=[]
    testing_data=[]
    testing_labels=[]
    for s,l in train:
        training_data.append(str(s.numpy()))
        training_labels.append(l.numpy())
    for s,l in test:
        testing_data.append(str(s.numpy()))
        testing_labels.append(l.numpy())
    tokeniser.fit_on_texts(training_data)
    training_data=convert_padded_tokens(training_data)
    testing_data=convert_padded_tokens(testing_data)
    training_labels=np.array(training_labels)
    testing_labels=np.array(testing_labels)
    return  training_data,training_labels,testing_data,testing_labels
training_data,training_labels,testing_data,testing_labels=get_data()
word_index=tokeniser.word_index
def build_model():
    model=tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_si,embedding_dim,input_length=max_length),
            tf.keras.layers.Conv1D(128,5,activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6,activation="relu"),
            tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()
    return model
model=build_model()
history=model.fit(training_data,training_labels,epochs=num_epochs, validation_data=(testing_data,testing_labels))
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
    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.title("loss")
    plt.show()
plot(history)
reverse_index=dict([(value,key) for (key,value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_index.get(i,'?') for i in text ])
