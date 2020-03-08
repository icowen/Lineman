import datetime
import os

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
    predict(model, test_plays_df)


def predict(model, test_plays_df):
    test_plays_add_1_to_all_x = test_plays_df.copy()
    # test_plays_add_1_to_all_x = test_plays_add_1_to_all_x.apply(lambda x: )
    test_plays_df["Predicted"] = model.predict(test_plays_df.drop(['PlayResult', 'Unnamed: 0', 'playId'], axis=1))
    plot_predictions(test_plays_df)


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


# main()
