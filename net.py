import datetime
import math
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

random.seed(1)
np.random.seed(2)
tf.random.set_seed(3)
tf.keras.backend.set_floatx('float32')

NUM_EPOCHS = 30
NUM_HIDDEN_NODES = 25
NUM_OUTPUT_NODES = 1
NUM_LAYERS = 2
BATCH_SIZE = 1000
MODEL = None
# MODEL = 'models/03-10-2020_20-36-47_epochs100_batch10188.h5'
DATA = None
DATA_FILENAME = 'fin_70.csv'
SAVE = True
KEEP_REGEX = r'(Off|OL|Def|frame|Match)'

TIME = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
FILENAME = f'{TIME}_epochs{NUM_EPOCHS}'
MODEL_SAVE_FILENAME = f'models/{FILENAME}.h5'
LOSS_PLOT_SAVE_FILENAME = f'loss_histories/{FILENAME}.png'


def main():
    global BATCH_SIZE, NUM_HIDDEN_NODES, NUM_EPOCHS
    # get_all_data()
    x_train_df = get_data_without_last_5_plays()
    x_test_df = DATA.loc[~DATA["playId"].isin(x_train_df["playId"].unique())]
    prior = 4
    model = get_model(x_train_df)
    initial_model = get_initial_net(model)
    update_net_to_use_prior(model, initial_model, x_train_df, prior)
    train_model(model, x_train_df)

    # result = predict(model, initial_model, prior, x_test_df)
    # print(result.to_string())

    for play_id in x_test_df["playId"].unique():
        for player_id in ["OL_C", "OL_LG", "OL_LT", "OL_RG", "OL_RT"]:
            fig, (ax1, ax2) = plt.subplots(2)
            get_rating_vs_frame_for_play_id(ax1, model, initial_model, x_test_df, prior, play_id, player_id, .01)
            get_S_vs_frame_graph_for_play(ax2, model, initial_model, x_test_df, prior, play_id)
            plt.tight_layout()
            plt.savefig(f'graphs/{TIME}_play{play_id}_player{player_id}.png')
            plt.close()


def get_data_without_last_5_plays():
    global DATA
    get_all_data()
    last_5_play_ids = DATA["playId"].unique()[-5:]
    df = DATA.copy()
    last_5_plays = df.loc[~df["playId"].isin(last_5_play_ids)]
    return last_5_plays


def get_all_data():
    global DATA
    data = pd.read_csv(DATA_FILENAME, dtype='float32')
    data.dropna(inplace=True)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    DATA = data
    x_train = DATA.copy()
    return x_train


def get_initial_net(model):
    initial_model = tf.keras.models.clone_model(model)
    initial_weights = model.get_weights()
    initial_model.set_weights(initial_weights)
    initial_model.save(f'models/{FILENAME}_initial.h5')
    return initial_model


def get_model(x_train):
    keep_cols = [c for c in x_train.columns if re.search(KEEP_REGEX, c)]
    input_shape = (x_train.drop([c for c in x_train.columns if c not in keep_cols], axis=1).shape[1],)
    output_shape = (len(x_train.index),)
    if MODEL:
        return tf.keras.models.load_model(MODEL)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN_NODES, input_shape=input_shape, activation=tf.nn.sigmoid))
    for i in range(NUM_LAYERS - 1):
        model.add(tf.keras.layers.Dense(NUM_HIDDEN_NODES, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(NUM_OUTPUT_NODES, activation=tf.keras.activations.linear))
    model.compile(optimizer='adam',
                  # loss=mse_loss_with_prior(K.placeholder(shape=output_shape, dtype='float32')),
                  loss=mse_loss_with_prior([]),
                  metrics=['acc'])

    return model


def train_model(model, df):
    keep_cols = [c for c in df.columns if re.search(KEEP_REGEX, c)]

    history = model.fit(df.drop([c for c in df.columns if c not in keep_cols], axis=1), df["PlayResult"],
                        validation_split=.2,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE)
    if SAVE:
        # plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True, expand_nested=True)
        # plot_loss(history)
        model.save(MODEL_SAVE_FILENAME)
    return model


def mse_loss_with_prior(avg_of_play_no_noise):
    def mse(y_true, y_pred):
        return K.mean(K.square((y_pred - avg_of_play_no_noise) - y_true))

    return mse


def update_net_to_use_prior(model, initial_net, x_train, prior=4):
    x_train = get_net_noise(initial_net, prior, x_train)
    model.compile(loss=mse_loss_with_prior(x_train["NetNoise"]))


def get_net_noise(initial_net, prior, df):
    df = df.copy()
    keep_cols = [c for c in df.columns if re.search(KEEP_REGEX, c)]
    initial_predictions = initial_net.predict(df.drop([c for c in df.columns if c not in keep_cols], axis=1))
    df["NetNoise"] = initial_predictions - prior
    return df


def predict(model, initial_model, prior, df, x_or_y=''):
    keep_cols = [c for c in df.columns if re.search(KEEP_REGEX, c)]
    label = f"Predicted{x_or_y}"
    df = get_net_noise(initial_model, prior, df)
    df.loc[:, label] = model.predict(df.drop([c for c in df.columns if c not in keep_cols], axis=1))
    df.loc[:, label] = df.apply(lambda x: x[label] - x["NetNoise"], axis=1)
    return df


def plot_predictions(test_plays_df):
    os.mkdir(f'pred_graphs/{FILENAME}')
    for n, grp in test_plays_df.groupby('playId'):
        file = f'pred_graphs/{FILENAME}/{n}.png'
        fig, ax = plt.subplots()
        plt.ylim(-20, 50)
        plt.xlabel('Frame ID')
        plt.ylabel('Yards')
        ax.scatter(x="frame.id", y="Predicted", data=grp, label="Predicted", c='b')
        ax.scatter(x="frame.id", y="PlayResult", data=grp, label="Actual", c='orange')
        ax.legend()
        plt.savefig(file)
    plt.show()


def get_rating_vs_frame_for_play_id(ax, model, initial_model, df, prior, play_id, player_label, delta):
    print(f'Generating rating vs frame for play {play_id}.')
    magnitudes = []
    play = df[df["playId"] == play_id]
    for frame_id in play["frame.id"].unique():
        frame_df = play[play["frame.id"] == frame_id]
        original_frame_df = predict(model, initial_model, prior, frame_df)

        move_x_df = frame_df.copy()
        move_x_df[f"{player_label}_x"] = move_x_df[f"{player_label}_x"].apply(lambda x: x + delta)
        move_x_df = predict(model, initial_model, prior, move_x_df, '_x')

        move_y_df = frame_df.copy()
        move_y_df[f"{player_label}_y"] = move_y_df[f"{player_label}_y"].apply(lambda x: x + delta)
        move_y_df = predict(model, initial_model, prior, move_y_df, '_y')

        dx = (original_frame_df.iloc[0]["Predicted"] - move_x_df.iloc[0]["Predicted_x"]) / delta
        dy = (original_frame_df.iloc[0]["Predicted"] - move_y_df.iloc[0]["Predicted_y"]) / delta
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        magnitudes.append(magnitude)
        ax.scatter(frame_id, magnitude, color='b')
    ax.set_title(f'Rating vs FrameId for Play {play_id} and Player {player_label}')
    ax.set_ylim(0, 1)
    plt.xlabel('Frame Id')
    plt.ylabel('Rating')


def get_S_vs_frame_graph_for_play(ax, model, initial_model, df, prior, play_id):
    print(f'Generating S vs Frame graph for play {play_id}.')
    df = df[df["playId"] == play_id]
    df = predict(model, initial_model, prior, df)
    ax.scatter(df["frame.id"], df["Predicted"], label='Predicted (ie. S)', c='red')
    ax.scatter(df["frame.id"], df["PlayResult"], label='Actual', c='blue')
    ax.set_title(f'S vs FrameId for Play {play_id}')
    ax.legend()
    plt.xlabel('Frame Id')
    plt.ylabel('Yards Gained (ie. S)')


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(LOSS_PLOT_SAVE_FILENAME)
    plt.close()


main()
