import math
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

np.set_printoptions(threshold=sys.maxsize)

data = pd.read_csv('train.csv')
keep_cols = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'Dis', 'Orientation', 'Dir', 'NflId', 'PossessionTeam',
             'NflIdRusher', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Yards']
data.drop([x for x in data.columns if x not in keep_cols], axis=1, inplace=True)
data["IsRusher"] = data.apply(lambda x: x['NflId'] == x['NflIdRusher'], axis=1)
data["IsOffense"] = data.apply(lambda x:
                               (x['Team'] == 'home') == (x['PossessionTeam'] == x['HomeTeamAbbr']), axis=1)

data = data.set_index(["GameId", "PlayId"])
data['OffenseX'] = data.loc[data['IsOffense'] & ~data['IsRusher']].groupby(['GameId', 'PlayId'])['X'].apply(list)
data['OffenseY'] = data.loc[data['IsOffense'] & ~data['IsRusher']].groupby(['GameId', 'PlayId'])['Y'].apply(list)
data['OffenseS'] = data.loc[data['IsOffense'] & ~data['IsRusher']].groupby(['GameId', 'PlayId'])['S'].apply(list)
data['RusherX'] = data.loc[data['IsRusher']].groupby(['GameId', 'PlayId'])['X'].first()
data['RusherY'] = data.loc[data['IsRusher']].groupby(['GameId', 'PlayId'])['Y'].first()
data['RusherS'] = data.loc[data['IsRusher']].groupby(['GameId', 'PlayId'])['S'].first()
data['DefenseX'] = data.loc[~data['IsOffense']].groupby(['GameId', 'PlayId'])['X'].apply(list)
data['DefenseY'] = data.loc[~data['IsOffense']].groupby(['GameId', 'PlayId'])['Y'].apply(list)
data['DefenseS'] = data.loc[~data['IsOffense']].groupby(['GameId', 'PlayId'])['S'].apply(list)
data = data.reset_index().drop_duplicates(subset=['GameId', 'PlayId'], keep='first')
data = data.set_index(["GameId", "PlayId"])

off_labels = [f'Off{i}' for i in range(10)]
def_labels = [f'Def{i}' for i in range(11)]
groups = data.groupby(['GameId', 'PlayId']).groups
index = pd.MultiIndex.from_product([groups, off_labels, def_labels], names=['Play', 'Off', 'Def'])
df = pd.DataFrame(index=index,
                  columns=['DefSpeed', 'DefRusherXY', 'DefRusherSpeed',
                           'OffDefXY', 'OffDefSpeed'])


def get_def_speed(x):
    i = int(re.search(r'[0-9]+', x.name[2])[0])
    return data.loc[x.name[0][0], x.name[0][1]]['DefenseS'][i]


def get_def_rusher_xy(x):
    i = int(re.search(r'[0-9]+', x.name[2])[0])
    rusher_x = data.loc[x.name[0][0], x.name[0][1]]['RusherX']
    rusher_y = data.loc[x.name[0][0], x.name[0][1]]['RusherY']
    def_x = data.loc[x.name[0][0], x.name[0][1]]['DefenseX'][i]
    def_y = data.loc[x.name[0][0], x.name[0][1]]['DefenseY'][i]
    return math.sqrt((rusher_x - def_x) ** 2 + (rusher_y - def_y) ** 2)


def get_def_rusher_speed(x):
    i = int(re.search(r'[0-9]+', x.name[2])[0])
    rusher_speed = data.loc[x.name[0][0], x.name[0][1]]['RusherS']
    def_speed = data.loc[x.name[0][0], x.name[0][1]]['DefenseS'][i]
    return def_speed - rusher_speed


def get_off_def_xy(x):
    def_i = int(re.search(r'[0-9]+', x.name[2])[0])
    off_i = int(re.search(r'[0-9]+', x.name[1])[0])
    def_x = data.loc[x.name[0][0], x.name[0][1]]['DefenseX'][def_i]
    off_x = data.loc[x.name[0][0], x.name[0][1]]['OffenseX'][off_i]
    def_y = data.loc[x.name[0][0], x.name[0][1]]['DefenseY'][def_i]
    off_y = data.loc[x.name[0][0], x.name[0][1]]['OffenseY'][off_i]
    return math.sqrt((def_x - off_x) ** 2 + (def_y - off_y) ** 2)


def get_off_def_speed(x):
    def_i = int(re.search(r'[0-9]+', x.name[2])[0])
    off_i = int(re.search(r'[0-9]+', x.name[1])[0])
    off_speed = data.loc[x.name[0][0], x.name[0][1]]['OffenseS'][off_i]
    def_speed = data.loc[x.name[0][0], x.name[0][1]]['DefenseS'][def_i]
    return off_speed - def_speed


df['DefSpeed'] = df.apply(get_def_speed, axis=1)
df['DefRusherXY'] = df.apply(get_def_rusher_xy, axis=1)
df['DefRusherSpeed'] = df.apply(get_def_rusher_speed, axis=1)
df['OffDefXY'] = df.apply(get_off_def_xy, axis=1)
df['OffDefSpeed'] = df.apply(get_off_def_speed, axis=1)


def convert_to_crps(yards):
    y = [0 for i in range(115)]
    for i in range(15 + yards, 115):
        y[i] = 1
    return np.pad(np.asarray(y), (84, 0), constant_values=0)


unstack = df.unstack(-1)
print(unstack[:10].to_string())
num_plays = len(df.index.get_level_values(0).unique())
train_data = np.reshape(unstack.to_numpy(),
                        (num_plays, 10, 5, 11))  # New dimensions = (num_plays, off, features, def)
train_values = np.asarray(
    list(map(np.asarray, data.groupby(['GameId', 'PlayId']).first()["Yards"].apply(convert_to_crps).values)))
print(train_data[0][0])  # Prints (off0, features, def)
num_test_plays = 10


def crps_loss_func(y_true, y_pred):
    ret = tf.where(y_true >= 1, y_pred - 1, y_pred)
    ret = K.square(ret)
    per_play_loss = K.sum(ret, axis=1)
    total_loss = K.mean(per_play_loss)
    return total_loss


def max_avg_pool_2D(x):
    # Possibly change 5 back to 10 if change data shape
    return tf.keras.layers.MaxPooling2D(pool_size=(1, 5))(x) * .3 \
           + tf.keras.layers.AveragePooling2D(pool_size=(1, 5))(x) * .7


def max_avg_pool_1D(x):
    # Possibly change 10 back to 11 if change data shape
    return tf.keras.layers.MaxPooling1D(pool_size=10)(x) * .3 \
           + tf.keras.layers.AveragePooling1D(pool_size=10)(x) * .7


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu', input_shape=train_data[0].shape))
model.add(tf.keras.layers.Conv2D(160, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(tf.keras.layers.Lambda(max_avg_pool_2D))
model.add(tf.keras.layers.Lambda(lambda x: K.squeeze(x, 2)))
model.add(tf.keras.layers.Conv1D(128, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(160, kernel_size=1, strides=1, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Lambda(max_avg_pool_1D))
model.add(tf.keras.layers.Lambda(lambda x: K.squeeze(x, 1)))
model.add(tf.keras.layers.Dense(96, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(199, activation='softmax'))

model.compile(optimizer='adam', loss=crps_loss_func, metrics=['accuracy'])
model.fit(train_data[:-num_test_plays], train_values[:-num_test_plays], epochs=50)

pred = model.predict(train_data[-num_test_plays:])
print(f'pred: {pred}')
print(f'actual: {train_values[-num_test_plays:] == 1}')
