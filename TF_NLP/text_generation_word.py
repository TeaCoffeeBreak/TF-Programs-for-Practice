import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
tokeniser= Tokenizer()
num_epochs=100
callback= tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, min_delta=0 ,mode="auto")
def get_data():
    file=open("Datasets/irish-lyrics-eof.txt","r").read()
    corpus=file.lower().split("\n")
    tokeniser.fit_on_texts(corpus)
    total_words= len(tokeniser.word_index)+1
    print(tokeniser.word_index)
    print(total_words)
    return corpus,total_words
corpus,total_words=get_data()
def preprocess(corpus):
    input_sequences=[]
    for line in corpus:
        token_list= tokeniser.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            ngram_sequence=token_list[:i+1]
            input_sequences.append(ngram_sequence)
    max_seq= max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq, padding="pre"))
    xs,labels= input_sequences[:,:-1], input_sequences[:,-1]
    ys= tf.keras.utils.to_categorical(labels,num_classes=total_words)
    return xs,ys,max_seq
xs,ys,max_seq=preprocess(corpus)

def build_model():
    model= tf.keras.models.Sequential([
            tf.keras.layers.Embedding(total_words,150, input_length=max_seq-1),
            # tf.keras.layers.Conv1D(512,5,activation="relu"),
            # tf.keras.layers.Conv1D(512,5,activation='relu'),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
        # tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(total_words,activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),loss="categorical_crossentropy",metrics=["acc"])
    print(model.summary())
    return  model
model=build_model()
history=model.fit(xs,ys,epochs=num_epochs,callbacks=[callback])
def plot(history,string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
plot(history,"acc")
plot(history,"loss")
def generate_text(seed_text,next_words):
    for i in range(next_words):
        token_list= tokeniser.texts_to_sequences([seed_text])[0]
        token_list= pad_sequences([token_list],maxlen=max_seq-1,padding="pre")
        predicted= model.predict_classes(token_list)
        output_word=""
        for word,index in tokeniser.word_index.items():
            if index == predicted:
                output_word=word
                break
        seed_text+=" "+output_word
    return seed_text
seed_text="I've got a bad feeling about this"
next_words=100
# print(generate_text(seed_text,next_words))
