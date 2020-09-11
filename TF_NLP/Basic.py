from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

sentences=["I love python",
           "I love C++"]

tokeniser= Tokenizer(num_words=100)
tokeniser.fit_on_texts(sentences)
word_index=tokeniser.word_index
print(word_index)
sequences= tokeniser.texts_to_sequences(sentences)
print(sequences)