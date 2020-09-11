from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tldextract
import numpy as  np
import pandas as pd
vocab_si=10000
embedding_dim=16
max_length=120
unique_value={}
trunc_type='post'
oov_token="<oov>"
num_epochs=10
sequence_length=None
n_char=None
def convert_to_tokens(train_data,val_data):
    global sequence_length,n_char
    tokenizer = Tokenizer(filters='', char_level=True, lower=False, oov_token=1)

    # fit only on training data
    tokenizer.fit_on_texts(train_data["url"])
    n_char = len(tokenizer.word_index.keys())

    train_seq = tokenizer.texts_to_sequences(train_data["url"])
    val_seq = tokenizer.texts_to_sequences(val_data["url"])

    sequence_length = np.array([len(i) for i in train_seq])
    sequence_length = np.percentile(sequence_length, 99).astype(int)
    train_seq = pad_sequences(train_seq, padding='post', maxlen=sequence_length)
    val_seq = pad_sequences(val_seq, padding='post', maxlen=sequence_length)

    return train_seq,val_seq
def get_data():
    data=pd.read_csv("/home/dj/Downloads/url_data/data.csv")
    val_size = 0.2
    train_data, val_data = train_test_split(data, test_size=val_size, stratify=data['label'], random_state=0)
    data = extract_url(data)
    train_data = extract_url(train_data)
    val_data = extract_url(val_data)

    for feature in ['subdomain', 'domain', 'domain_suffix']:
        # get unique value
        label_index = {label: index for index, label in enumerate(train_data[feature].unique())}

        # add unknown label in last index
        label_index['<unknown>'] = list(label_index.values())[-1] + 1

        # count unique value
        unique_value[feature] = label_index['<unknown>']

        # encode
        train_data.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in
                                      train_data.loc[:, feature]]
        val_data.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in
                                    val_data.loc[:, feature]]

    for data in [train_data, val_data]:
        data.loc[:, 'label'] = [0 if i == 'good' else 1 for i in data.loc[:, 'label']]
    print(train_data.head())
    return train_data,val_data


def parsed_url(url):
    # extract subdomain, domain, and domain suffix from url
    # if item == '', fill with '<empty>'
    subdomain, domain, domain_suffix = ('<empty>' if extracted == '' else extracted for extracted in
                                        tldextract.extract(url))

    return [subdomain, domain, domain_suffix]


def extract_url(data):
    # parsed url
    extract_url_data = [parsed_url(url) for url in data['url']]
    extract_url_data = pd.DataFrame(extract_url_data, columns=['subdomain', 'domain', 'domain_suffix'])

    # concat extracted feature with main data
    data = data.reset_index(drop=True)
    data = pd.concat([data, extract_url_data], axis=1)

    return data

train_data,val_data=get_data()
train_seq,val_seq=convert_to_tokens(train_data,val_data)


def convolution_block(x):
    conv_3_layer = tf.keras.layers.Conv1D(64, 3, padding='same', activation='elu')(x)
    conv_5_layer = tf.keras.layers.Conv1D(64, 5, padding='same', activation='elu')(x)
    conv_layer = tf.keras.layers.concatenate([x, conv_3_layer, conv_5_layer])
    conv_layer = tf.keras.layers.Flatten()(conv_layer)
    return conv_layer


def embedding_block(unique_value, size, name):
    input_layer = tf.keras.layers.Input(shape=(1,), name=name + '_input')
    embedding_layer = tf.keras.layers.Embedding(unique_value, size, input_length=1)(input_layer)
    return input_layer, embedding_layer


def create_model(sequence_length, n_char, unique_value):
    input_layer = []

    # sequence input layer
    sequence_input_layer = tf.keras.layers.Input(shape=(sequence_length,), name='url_input')
    input_layer.append(sequence_input_layer)

    # convolution block
    char_embedding = tf.keras.layers.Embedding(n_char + 1, 32, input_length=sequence_length)(sequence_input_layer)
    conv_layer = convolution_block(char_embedding)

    # entity embedding
    entity_embedding = []
    for key, n in unique_value.items():
        size = 4
        input_l, embedding_l = embedding_block(n + 1, size, key)
        embedding_l = tf.keras.layers.Reshape(target_shape=(size,))(embedding_l)
        input_layer.append(input_l)
        entity_embedding.append(embedding_l)

    # concat all layer
    fc_layer = tf.keras.layers.concatenate([conv_layer, *entity_embedding])
    fc_layer = tf.keras.layers.Dropout(rate=0.5)(fc_layer)

    # dense layer
    fc_layer = tf.keras.layers.Dense(128, activation='elu')(fc_layer)
    fc_layer = tf.keras.layers.Dropout(rate=0.2)(fc_layer)

    # output layer
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(fc_layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


# create model
model = create_model(sequence_length, n_char, unique_value)
train_x = [train_seq, train_data['subdomain'], train_data['domain'], train_data['domain_suffix']]
train_y = train_data['label'].values

# model training
early_stopping = [tf.keras.callbacks.EarlyStopping(monitor='val_precision', patience=5, restore_best_weights=True, mode='max')]
history = model.fit(train_x, train_y, batch_size=64, epochs=25, verbose=1, validation_split=0.2, shuffle=True, callbacks=early_stopping)
model.save('model.h5')
