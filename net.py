import datetime
import math
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.utils import plot_model

NUM_EPOCHS = 200
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = 1
BATCH_SIZE = 10
NUM_TEST_SAMPLES = 500
LOSS_FUNCTION = 'mse'
# MODEL = None
MODEL = 'models/02-24-2020_17-01-52_epochs200_batch10.h5'
X, Y = None, None
SAVE = True

time = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
FILENAME = f'{time}_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}'
MODEL_SAVE_FILENAME = f'models/{FILENAME}.h5'
LOSS_PLOT_SAVE_FILENAME = f'loss_histories/{FILENAME}.png'


def main():
    global X, Y
    X, Y, test_plays_df = get_data()

    model = get_model()
    data = pd.read_csv('tracking_data.csv')
    y = data['PlayResult']
    play_id = 1317
    # play_id = data["playId"].unique()[0]
    play = data[data['playId'] == play_id]
    predict(model, play, '')
    # magnitudes = []
    # fig, ax = plt.subplots()
    # for frame in play["frame.id"].unique():
    #     test_plays_df = play[play["frame.id"] == frame]
    #     move_x = test_plays_df.copy()
    #     move_y = test_plays_df.copy()
    #     test_plays_df = predict(model, test_plays_df, '')
    #     move_x["OL_1_x"] = move_x["OL_1_x"].apply(lambda x: x+.01)
    #     move_x = predict(model, move_x, '_x')
    #     move_y["OL_1_y"] = move_y["OL_1_y"].apply(lambda x: x+.01)
    #     move_y = predict(model, move_y, '_y')
    #     dx = (test_plays_df.iloc[0]["Predicted"] - move_x.iloc[0]["Predicted_x"]) / .01
    #     dy = (test_plays_df.iloc[0]["Predicted"] - move_y.iloc[0]["Predicted_y"]) / .01
    #     magnitude = math.sqrt(dx ** 2 + dy ** 2)
    #     print(f'frame: {frame} dx: {dx} dy: {dy} magnitude: {magnitude}')
    #     magnitudes.append(magnitude)
    #     ax.scatter(frame, magnitude)
    # print(f'magnitudes: {magnitudes}')
    # plt.show()
''' Try using prior of 4 yards every play'''

def predict(model, test_plays_df, x_or_y):
    test_plays_add_1_to_all_x = test_plays_df.copy()
    # test_plays_add_1_to_all_x = test_plays_add_1_to_all_x.apply(lambda x: )
    if 'PlayResult' in test_plays_df.columns:
        test_plays_df = test_plays_df.drop(['PlayResult', 'Unnamed: 0', 'playId'], axis=1)
    test_plays_df[f"Predicted{x_or_y}"] = model.predict(test_plays_df)
    # print(test_plays_df.to_string())
    fig, ax = plt.subplots()
    ax.scatter(test_plays_df["frame.id"], test_plays_df["Predicted"])
    plt.show()
    return test_plays_df
    # plot_predictions(test_plays_df)


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


def get_data():
    x = pd.read_csv('tracking_data.csv')
    x.dropna(inplace=True)
    last_5_play_ids = x["playId"].unique()[-5:]
    last_5_plays_df = x.loc[x["playId"].isin(last_5_play_ids)]
    x = x.loc[~x["playId"].isin(last_5_play_ids)]   # Remove last 5 plays
    y = x['PlayResult']
    x.drop(['PlayResult', 'Unnamed: 0', 'playId'], axis=1, inplace=True)
    x = x.values
    y = y.values
    return x, y, last_5_plays_df


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


def train_model(model, x, y):
    history = model.fit(x, y,
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


def get_model():
    if MODEL:
        return tf.keras.models.load_model(MODEL)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN_NODES, input_shape=(X[0].shape[0],), activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(NUM_OUTPUT_NODES, activation=tf.keras.activations.linear))
    model.compile(optimizer='adam',
                  loss=LOSS_FUNCTION,
                  metrics=['acc'])
    train_model(model, X, Y)
    return model


main()
