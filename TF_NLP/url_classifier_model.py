from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tldextract
import numpy as  np
import pandas as pd

unique_value = {}
tokenizer = None
sequence_length = None
n_char = None
word_index = None


def convert_to_tokens(train_data, val_data):
    global tokenizer, sequence_length, n_char, word_index
    tokenizer = Tokenizer(filters='', char_level=True, lower=False, oov_token=1)

    # fit only on training data
    tokenizer.fit_on_texts(train_data["url"])
    n_char = len(tokenizer.word_index.keys())
    word_index = tokenizer.word_index
    train_seq = tokenizer.texts_to_sequences(train_data["url"])
    val_seq = tokenizer.texts_to_sequences(val_data["url"])

    sequence_length = np.array([len(i) for i in train_seq])
    sequence_length = np.percentile(sequence_length, 99).astype(int)
    train_seq = pad_sequences(train_seq, padding='post', maxlen=sequence_length)
    val_seq = pad_sequences(val_seq, padding='post', maxlen=sequence_length)

    return train_seq, val_seq


def get_data():
    data = pd.read_csv("Datasets/malicious_url_detect/data.csv")
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
    return train_data, val_data


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


train_data, val_data = get_data()
train_seq, val_seq = convert_to_tokens(train_data, val_data)


def load_model():
    return tf.keras.models.load_model("/home/dj/Downloads/malicious_url_detect/model.h5")


model = load_model()


def malicious_url(urls, model):
    data_val = [parsed_url(u) for u in urls]
    data_val = pd.DataFrame(data_val, columns=['subdomain', 'domain', 'domain_suffix'])
    print(data_val)
    for feature in ['subdomain', 'domain', 'domain_suffix']:
        # get unique value
        label_index = {label: index for index, label in enumerate(train_data[feature].unique())}

        # add unknown label in last index
        label_index['<unknown>'] = list(label_index.values())[-1] + 1

        # count unique value
        # unique_value[feature] = label_index['<unknown>']

        # encode
        data_val.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in
                                    data_val.loc[:, feature]]

    # print(data_val)
    data_seq = pad_sequences(tokenizer.texts_to_sequences(urls), padding='post', maxlen=sequence_length)
    val_x = [data_seq, data_val['subdomain'], data_val['domain'], data_val['domain_suffix']]
    val_pred = model.predict(val_x)
    # print(val_pred)
    val_pred = np.where(val_pred[:, 0] >= 0.7, 1, 0)
    val_pred = ["Good" if v == 1 else "Bad" for v in val_pred]
    return val_pred


urls = ["www.google.com", "diaryofagameaddict.com", "titon.info"]
prediction = malicious_url(urls, model)

print("Prediction ",prediction)