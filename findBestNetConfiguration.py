import datetime
import sys
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

random.seed(1)
np.random.seed(2)
tf.random.set_seed(3)
tf.keras.backend.set_floatx('float32')


DATA_FILENAME = 'fin_70.csv'
KEEP_REGEX = r'(Off|OL|Def|frame|Match)'


def main():
    x_train_df = get_data_without_last_5_plays()
    prior = 4
    num_epochs = 100
    best_loss = {"loss": sys.maxsize}
    for num_layers in range(1, 3):
        for num_hidden_nodes in range(5, 100, 5):
            print(f'Building net with {num_layers} layers and {num_hidden_nodes} nodes.')
            time = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
            filename = f'{time}_layers{num_layers}_nodes{num_hidden_nodes}'
            model_filename = f'models/{filename}.h5'
            loss_filename = f'loss_histories/{filename}.png'
            model = get_model(x_train_df, num_layers, num_hidden_nodes)
            initial_model = get_initial_net(model, filename)
            update_net_to_use_prior(model, initial_model, x_train_df, prior)
            model, history = train_model(model, x_train_df, num_epochs, model_filename, loss_filename)
            val_loss = history.history['val_loss']
            min_val_loss = min(val_loss)
            min_val_loss_index = val_loss.index(min_val_loss)
            if min_val_loss < best_loss['loss']:
                best_loss['loss'] = min_val_loss
                best_loss['epoch'] = min_val_loss_index
                best_loss['num_hidden_nodes'] = num_hidden_nodes
                best_loss['num_layers'] = num_layers
                best_loss['model_filename'] = model_filename
                best_loss['loss_filename'] = loss_filename
            print(best_loss)
    print(best_loss)


def get_data_without_last_5_plays():
    x_train = get_all_data()
    last_5_play_ids = x_train["playId"].unique()[-5:]
    last_5_plays = x_train.loc[~x_train["playId"].isin(last_5_play_ids)]
    return last_5_plays


def get_all_data():
    data = pd.read_csv(DATA_FILENAME, dtype='float32')
    data.dropna(inplace=True)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    x_train = data.copy()
    return x_train


def get_initial_net(model, model_filename):
    initial_model = tf.keras.models.clone_model(model)
    initial_weights = model.get_weights()
    initial_model.set_weights(initial_weights)
    initial_model.save(f'models/{model_filename}_initial.h5')
    return initial_model


def get_model(x_train, num__hidden_layers, num_hidden_nodes):
    keep_cols = [c for c in x_train.columns if re.search(KEEP_REGEX, c)]
    input_shape = (x_train.drop([c for c in x_train.columns if c not in keep_cols], axis=1).shape[1],)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_hidden_nodes, input_shape=input_shape, activation=tf.nn.sigmoid))

    for layer in range(num__hidden_layers - 1):
        model.add(tf.keras.layers.Dense(num_hidden_nodes, activation=tf.keras.activations.linear))

    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))
    model.compile(optimizer='adam',
                  loss=mse_loss_with_prior([]),
                  metrics=['acc'])

    return model


def train_model(model, df, num_epochs, model_filename, loss_filename):
    keep_cols = [c for c in df.columns if re.search(KEEP_REGEX, c)]

    history = model.fit(df.drop([c for c in df.columns if c not in keep_cols], axis=1), df["PlayResult"],
                        validation_split=.2,
                        epochs=num_epochs,
                        batch_size=100)
    plot_loss(history, loss_filename)
    model.save(model_filename)
    return model, history


def mse_loss_with_prior(avg_of_play_no_noise):
    def mse(y_true, y_pred):
        return K.mean(K.square((y_pred - avg_of_play_no_noise) - y_true))

    return mse


def update_net_to_use_prior(model, initial_net, x_train, prior):
    x_train = get_net_noise(initial_net, prior, x_train)
    model.compile(loss=mse_loss_with_prior(x_train["NetNoise"]))


def get_net_noise(initial_net, prior, df):
    df = df.copy()
    keep_cols = [c for c in df.columns if re.search(KEEP_REGEX, c)]
    initial_predictions = initial_net.predict(df.drop([c for c in df.columns if c not in keep_cols], axis=1))
    df["NetNoise"] = initial_predictions - prior
    return df


def plot_loss(history, loss_filename):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(loss_filename)
    plt.close()


main()
