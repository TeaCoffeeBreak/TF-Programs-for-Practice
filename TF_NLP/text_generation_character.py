import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
seq_length=100
examples_per_epoch=None
BATCH_SI=64
BUFFER=10000
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# path_to_file="/home/dj/Downloads/irish-lyrics-eof.txt"
ck_dir= "/"
ck_prefix=os.path.join(ck_dir,"ckpt_{epoch}")
ckpt_callback=tf.keras.callbacks.ModelCheckpoint(filepath=ck_prefix, save_weights_only=True)
def get_data():
    global examples_per_epoch
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    print("length of text:{}".format(len(text)))
    #unique characters in text
    vocab = sorted(set(text))
    print("{} unique characters in text".format(len(vocab)))
    #creating unique charater to indices mappping
    char2idx={u:i for i,u in enumerate(vocab)}
    idx2chax=np.array(vocab)
    text_as_int= np.array([char2idx[i] for i in text])
    print("{} ---character mapped to int ---> {}".format(repr(text[:10]),text_as_int[:10]))

    examples_per_epoch=len(text)//(seq_length+1)
    char_dataset= tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences= char_dataset.batch(seq_length+1,drop_remainder=True)
    return sequences,vocab,char2idx,idx2chax
sequences,vocab,char2idx,idx2char=get_data()
def split_input(chunks):
    input_text=chunks[:-1]
    target_text=chunks[1:]
    return input_text,target_text
dataset=sequences.map(split_input)

def shuffle_dataset(dataset):
    return dataset.shuffle(BUFFER).batch(BATCH_SI, drop_remainder=True)
dataset=shuffle_dataset(dataset)
vocab_si=len(vocab)
embedding_dim=256
num_epochs=10
def build_model(BATCH_SI):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_si,embedding_dim,batch_input_shape=[BATCH_SI,None]),
      # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True,
      #                   stateful=True,
      #                   recurrent_initializer='glorot_uniform')),
     # tf.keras.layers.Bidirectional( tf.keras.layers.GRU(1024,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform')),
        tf.keras.layers.GRU(1024,return_sequences=True,stateful=True,recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dense(vocab_si)
    ])

    return model
model=build_model(64)
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["acc"])
history=model.fit(dataset,epochs=num_epochs,callbacks=[ckpt_callback])

model=build_model(1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir=ck_dir))
model.build(tf.TensorShape([1,None]))
print(model.summary())
def plot(history,string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

#
plot(history, "acc")
plot(history, "loss")

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      # predictions = model.predict_classes(input_eval)
      predictions=model(input_eval)
      # remove the batch dimension
      # print(predictions)
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      # predicted_id=[idx2char[i] for i in predictions]
      # print(predicted_id)
      text_generated.append(idx2char[predicted_id])
      # input_eval=tf.expand_dims([predictions[-1]],0)
  return (start_string + ''.join(text_generated))
print(generate_text(model,u"I have a bad feeling about this"))