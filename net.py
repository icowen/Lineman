import datetime

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.utils import plot_model

NUM_EPOCHS = 50
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = 1
BATCH_SIZE = 10
NUM_TEST_SAMPLES = 500
LOSS_FUNCTION = 'mse'
X, Y = None, None
SAVE = True

time = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
filename = f'{time}_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}.h5'
MODEL_SAVE_FILENAME = f'models/{filename}'
LOSS_PLOT_SAVE_FILENAME = f'loss_histories/{filename}.png'


def main():
    global X, Y, LOSS_FUNCTION
    X, Y, test_x, test_y = get_data()

    x_train = X
    y_train = Y
    model = get_model()
    train_model(model, x_train, y_train)
    predict(model, test_x, test_y)


def predict(model, x_test, y_test):
    prediction = model.predict(x_test)
    for p, a in zip(prediction, y_test):
        print(f'prediction: {p};\tActual: {a}')


def get_data():
    x = pd.read_csv('tracking_data.csv')
    x = x.dropna()
    test_x = x.loc[x["playId"] == 4805]
    test_x = test_x.drop(['PlayResult', 'Unnamed: 0', 'playId'], axis=1)
    test_y = x.loc[x["playId"] == 4805]['PlayResult']
    y = x['PlayResult']
    x = x.drop(['PlayResult', 'Unnamed: 0', 'playId'], axis=1)
    x = x.values
    y = y.values
    # kf = KFold(n_splits=2)
    # for train, test in kf.split(X):
    #     print("%s %s" % (train, test))
    return x, y, test_x, test_y


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(LOSS_PLOT_SAVE_FILENAME)


def train_model(model, x, y, use_callbacks=False):
    validation_overfitting = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=1,
                                                              patience=50,
                                                              verbose=0,
                                                              mode='min')
    callbacks_list = []
    if use_callbacks:
        callbacks_list.append(validation_overfitting)

    history = model.fit(x, y,
                        # validation_split=.2,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks_list
                        )
    plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True, expand_nested=True)
    if SAVE:
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
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(NUM_HIDDEN_NODES, input_shape=(X[0].shape[0],), activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(NUM_OUTPUT_NODES, activation=tf.keras.activations.linear))
    model.compile(optimizer='adam',
                  loss=LOSS_FUNCTION,
                  metrics=['acc'])
    return model


main()
