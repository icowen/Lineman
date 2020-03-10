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
from tensorflow.keras.utils import plot_model

random.seed(1)
np.random.seed(2)
tf.random.set_seed(3)
tf.keras.backend.set_floatx('float64')

NUM_EPOCHS = 1
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = 1
BATCH_SIZE = 10188
NUM_TEST_SAMPLES = 500
LOSS_FUNCTION = 'mse'
MODEL = None
# MODEL = 'models/all_plays_get_4_yards.h5'
X, Y = None, None
DATA = None
SAVE = True

TIME = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
FILENAME = f'{TIME}_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}'
MODEL_SAVE_FILENAME = f'models/{FILENAME}.h5'
LOSS_PLOT_SAVE_FILENAME = f'loss_histories/{FILENAME}.png'


def main():
    global BATCH_SIZE
    # get_all_data()
    x_train_df = get_data_without_last_5_plays()
    model = get_model(x_train_df)
    update_net_to_use_prior(model, x_train_df)
    predict(model, x_train_df)

    print(x_train_df.head().to_string())
    train_model(model, x_train_df)
    result = predict(model, x_train_df)
    print(result.head().to_string())

    # DATA["Prediction"] = model.predict(DATA.drop(["PlayResult", "playId", "NetNoise", "Prediction"], axis=1))
    # DATA["Prediction"] = DATA.apply(lambda x: x["Prediction"] - x["NetNoise"], axis=1)
    # print(DATA.head().to_string())
    # play_id = DATA["playId"].unique()[-2]
    # get_S_vs_frame_graph_for_play(model, DATA, play_id)
    # get_rating_vs_frame_for_play_id(model, DATA, play_id, .01)


def update_net_to_use_prior(model, x_train, prior=4):
    keep_cols = [c for c in x_train.columns if re.search(r'(Off|OL|Def|frame)', c)]

    initial_net = tf.keras.models.clone_model(model)
    initial_weights = model.get_weights()
    initial_net.set_weights(initial_weights)
    initial_predictions = initial_net.predict(x_train.drop([c for c in x_train.columns if c not in keep_cols], axis=1))
    x_train["NetNoise"] = initial_predictions - prior
    model.compile(loss=mse_loss_with_prior(x_train["NetNoise"]))


''' Try using prior of 4 yards every play'''


def get_rating_vs_frame_for_play_id(model, df, play_id, delta):
    print(f'Generating rating vs frame for play {play_id}.')
    magnitudes = []
    play = df[df["playId"] == play_id]
    fig, ax = plt.subplots()
    for frame_id in play["frame.id"].unique():
        frame = play[play["frame.id"] == frame_id]
        original_frame = predict(model, frame)

        move_x = frame.copy()
        move_x["OL_1_x"] = move_x["OL_1_x"].apply(lambda x: x + delta)
        move_x = predict(model, move_x, '_x')

        move_y = frame.copy()
        move_y["OL_1_y"] = move_y["OL_1_y"].apply(lambda x: x + delta)
        move_y = predict(model, move_y, '_y')

        dx = (original_frame.iloc[0]["Predicted"] - move_x.iloc[0]["Predicted_x"]) / delta
        dy = (original_frame.iloc[0]["Predicted"] - move_y.iloc[0]["Predicted_y"]) / delta
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        magnitudes.append(magnitude)
        ax.scatter(frame_id, magnitude, color='b')
    print(f'magnitudes: {magnitudes}')
    ax.set_title(f'Rating vs FrameId for Play {play_id}')
    plt.xlabel('Frame Id')
    plt.ylabel('Rating')
    plt.savefig(f'rating_vs_frame_graphs/{TIME}_play{play_id}_delta{delta}.png')


def predict(model, df, x_or_y=''):
    keep_cols = [c for c in df.columns if re.search(r'(Off|OL|Def|frame)', c)]
    label = f"Predicted{x_or_y}"
    df[label] = model.predict(df.drop([c for c in df.columns if c not in keep_cols], axis=1))
    df[label] = df.apply(lambda x: x[label] - x["NetNoise"], axis=1)
    return df


def get_S_vs_frame_graph_for_play(model, df, play_id):
    print(f'Generating S vs Frame graph for play {play_id}.')
    df = df[df["playId"] == play_id]
    if 'PlayResult' in df.columns:
        df = df.drop(['PlayResult', 'playId'], axis=1)
    df[f"Predicted"] = model.predict(df)
    fig, ax = plt.subplots()
    ax.scatter(df["frame.id"], df["Predicted"])
    ax.set_title(f'S vs FrameId for Play {play_id}')
    plt.xlabel('Frame Id')
    plt.ylabel('Predicted Yards Gained (ie. S)')
    plt.savefig(f'predicted_vs_frame_graphs/{TIME}_play{play_id}.png')


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


def get_all_data():
    global DATA
    data = pd.read_csv('tracking_data.csv')
    data.dropna(inplace=True)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    DATA = data
    x_train = DATA.copy()
    return x_train


def get_data_without_last_5_plays():
    global DATA
    get_all_data()
    last_5_play_ids = DATA["playId"].unique()[-5:]
    DATA = DATA.loc[~DATA["playId"].isin(last_5_play_ids)]
    x_train = DATA.copy()
    return x_train


def move_players(play_id, frame_id, df, ol_x_shift=0, ol_y_shift=0):
    def draw_football_field(ax):
        ax.set_xlim([0, 120])
        ax.set_ylim([0, 160 / 3])
        ax.set_facecolor('green')
        ax.fill_between([0, 10], 160 / 3, color='maroon')
        ax.fill_between([110, 120], 160 / 3, color='gold')

        for i in range(1, 12):
            ax.axvline(i * 10, c='white', zorder=0)
            if i < 6:
                ax.annotate(i * 10, ((i + 1) * 10 + 1, 5), c='white')
            elif i < 10:
                ax.annotate(100 - i * 10, ((i + 1) * 10 + 1, 5), c='white')

    play = df[df["playId"] == play_id]
    frames = play["frame.id"].values
    print(f'Number of frames for play {play_id}: {frames.size}')
    if frame_id not in frames:
        print(f'Frame id {frame_id} not a valid frame id for play {play_id}.')
        return None
    frame_1 = play[play["frame.id"] == frame_id]
    x_cols = [c for c in frame_1.columns if '_x' in c.lower()]
    y_cols = [c for c in frame_1.columns if '_y' in c.lower()]
    points = [(frame_1[x][0], frame_1[y][0], re.findall(r'^(.*?)_', x.lower())[0]) for x, y in zip(x_cols, y_cols)]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    label = [p[2] for p in points]
    df = pd.DataFrame(dict(x=x, y=y, label=label))
    groups = df.groupby('label')

    fig, axs = plt.subplots(2, figsize=(6, 8), sharex=True, sharey=True)
    for ax in axs:
        draw_football_field(ax)

    marker_size = 16
    edge_color = 'black'

    for name, group in groups:
        if name == 'def':
            m = 's'
        else:
            m = 'o'

        if name == 'ol':
            axs[1].scatter(group.x, group.y, label=f'{name}_old',
                           marker=m, edgecolors=edge_color, color='gray',
                           s=marker_size)
            axs[1].scatter(group.x + ol_x_shift, group.y + ol_y_shift, label=name,
                           edgecolors=edge_color, marker=m, s=marker_size)
        else:
            axs[1].scatter(group.x, group.y, label=name,
                           marker=m, s=marker_size, edgecolors=edge_color)

        axs[0].scatter(group.x, group.y, label=name, marker=m,
                       s=marker_size, edgecolors=edge_color)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    axs[0].title.set_text('Original Positions')
    axs[1].title.set_text('New Positions')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('C:/Users/ian_c/Linesman/football_fields/ol_plus_2_in_y_direction.svg')


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(LOSS_PLOT_SAVE_FILENAME)


def train_model(model, df):
    global BATCH_SIZE
    BATCH_SIZE = len(df.index)
    keep_cols = [c for c in df.columns if re.search(r'(Off|OL|Def|frame)', c)]

    history = model.fit(df.drop([c for c in df.columns if c not in keep_cols], axis=1), df["PlayResult"],
                        validation_split=.2,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE)
    if SAVE:
        plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True, expand_nested=True)
        plot_loss(history)
        model.save(MODEL_SAVE_FILENAME)
    return model


def crps_loss_with_prior(avg_of_play_no_noise):
    def crps(y_true, y_pred):
        logit_of_y_pred = K.log(y_pred / (1 - K.clip(y_pred, 0, 1 - 10 ** -16)))
        sum_of_logits = avg_of_play_no_noise + logit_of_y_pred
        inverse_logit = K.exp(sum_of_logits) / (1 + K.exp(sum_of_logits))
        inverse_logit = tf.where(tf.math.is_nan(inverse_logit), tf.ones_like(inverse_logit), inverse_logit)
        ret = tf.where(y_true >= 1, inverse_logit - 1, inverse_logit)
        ret = K.square(ret)
        per_play_loss = K.sum(ret, axis=1)
        total_loss = K.mean(per_play_loss)
        return total_loss

    return crps


def mse_loss_with_prior(avg_of_play_no_noise):
    def mse(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.float64)
        return K.mean(K.square((y_pred - avg_of_play_no_noise) + K.square(y_true)))

    return mse


def get_model(x_train):
    keep_cols = [c for c in x_train.columns if re.search(r'(Off|OL|Def|frame)', c)]
    input_shape = (x_train.drop([c for c in x_train.columns if c not in keep_cols], axis=1).shape[1], )
    output_shape = (len(x_train.index), )
    if MODEL:
        return tf.keras.models.load_model(MODEL)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN_NODES, input_shape=input_shape, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(NUM_OUTPUT_NODES, activation=tf.keras.activations.linear))
    model.compile(optimizer='adam',
                  # loss=LOSS_FUNCTION,
                  loss=mse_loss_with_prior(K.placeholder(shape=output_shape, dtype='float64')),
                  metrics=['acc'])

    return model


main()
