import re
import string
import numpy as np
import pandas as pd
import contractions
import tensorflow as tf
import nltk
from tensorflow import keras
from tensorflow.keras import layers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, GlobalAvgPool1D
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import tensorflow_hub as hub
import tensorflow_text as text

'''tensorflow.python.framework.errors_impl.AlreadyExistsError: Another metric with the same name already exists.
Keep tensorflow==2.6.0 and keras==2.6.0'''

matplotlib.rcParams['figure.dpi'] = 150

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # self-attention layer
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # layer norm
        ffn_output = self.ffn(out1)  #feed-forward layer
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # layer norm

def text_preprocessing(text, punc_to_remove, english_stopwords):

    # print(text)
    # Lowercase
    text = text.lower()

    # Removing contractions: I'd = I would  , I'll = I will
    text = re.sub(r"(`)+", "'", text)
    text = contractions.fix(text)

    # Removing Punctuations(symbols): maketrans will map the symbols to its replacement and then translate
    text = text.translate(str.maketrans("","", punc_to_remove))

    # Removing digits
    text = "".join([ch for ch in text if not ch.isdigit()])

    # Removing multiple spaces
    text = " ".join(text.split())

    # Removing stopwords
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in english_stopwords]
    text = " ".join(tokens)


    return text

# Plotting results
def plot1(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    ## Accuracy plot
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    ## Loss plot
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot2(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0,1)
    plt.show()


if __name__ == '__main__':

    Bert_training_mode = "off"

    #Loading the kaggle sentiment analysis dataset
    data = pd.read_csv(r"C:\Users\703294213\Documents\Work\Project Work\My Projects\Sentiment Analysis dataset\data\train.csv",encoding='unicode_escape')
    data = data[['selected_text', 'sentiment']]

    # Handling any null values
    data.dropna(subset=['selected_text', 'sentiment'], inplace=True)

    punc_to_remove = string.punctuation
    nltk.download('stopwords')
    nltk.download('punkt')

    english_stopwords = stopwords.words('english')


    data['selected_text'] = data['selected_text'].apply(lambda x: text_preprocessing(x, punc_to_remove, english_stopwords))

    # Remove those records which got blank after text cleaning
    data['selected_text'].replace("", np.nan, inplace=True)
    data.dropna(subset=['selected_text'], inplace=True)

    '''Splitting the data into train and test set'''
    train_set, test_set = train_test_split(data, test_size = 0.1, random_state = 42)

    # Hyperparameter for tokenizer
    vacab_size = 1000

    '''Train set'''
    texts = np.array(train_set['selected_text'])
    label = []
    for out in list(train_set['sentiment']):
        if out == 'positive':
            label.append(0)
        elif out == 'negative':
            label.append(1)
        else:
            label.append(2)
    label = np.array(label)

    max_length = 0
    for text in texts:
        text_length = len(text.split(" "))
        if max_length < text_length:
            max_length = text_length

    x_train = texts
    y_train = label


    '''Test set'''
    texts = np.array(test_set['selected_text'])
    label = []
    for out in list(test_set['sentiment']):
        if out == 'positive':
            label.append(0)
        elif out == 'negative':
            label.append(1)
        else:
            label.append(2)
    label = np.array(label)

    x_test = texts
    y_test = label

    preprocessor_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2"

    if Bert_training_mode == "off":

        '''When using bert model directly'''
        preprocess = hub.load(preprocessor_url)
        encoder = hub.load(bert_url)

        # Use BERT on a batch of raw text for embedding inputs.
        input_train = preprocess(x_train[0:5000])
        pooled_output_train = encoder(input_train)["pooled_output"]
        y_train = y_train[0:5000]
        x_train = pooled_output_train

        input_test = preprocess(x_test[0:500])
        pooled_output_test = encoder(input_test)["pooled_output"]
        y_test = y_test[0:500]
        x_test = pooled_output_test


        # Building Neural network using Functional API
        input_layer = []
        output_layer = []

        text_input = Input(shape=(256, ), dtype=tf.float32)
        dense = Dense(64, activation="relu")(text_input)
        dense = Dropout(0.1)(dense)
        dense = Dense(32, activation="relu")(dense)
        dense = Dropout(0.1)(dense)
        output = Dense(3, activation="softmax")(dense)

        input_layer.append(text_input)
        output_layer.append(output)
        model = Model(input_layer, output_layer)

    else:

        '''When using BERT model with trainable parameters'''

        input_layer = []
        output_layer = []

        text_input = Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(preprocessor_url)
        encoder_input = preprocessor(text_input)
        encoder = hub.KerasLayer(bert_url, trainable=True)
        outputs = encoder(encoder_input)
        pooled_output = outputs['pooled_output']

        dense = Dense(64, activation="relu")(pooled_output)
        dense = Dropout(0.1)(dense)
        dense = Dense(32, activation="relu")(dense)
        dense = Dropout(0.1)(dense)
        output = Dense(3, activation="softmax")(dense)

        input_layer.append(text_input)
        output_layer.append(output)
        model = Model(input_layer, output_layer)


    model.compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"]
                  )
    history = model.fit(x_train,
              y_train,
              batch_size = 16,
              epochs = 100,
              validation_data = (x_test, y_test))

    plot1(history)

    print("Done")
