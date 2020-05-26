import datetime
import math
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def main():
    # data = get_data()
    # data = standardized_play_direction(data)
    # data = combine_by_team(data)
    # data = data[22977:]
    # df = create_features(data)
    # train_data, train_values = get_test_train_data(data, df)
    # for i in range(32):
    #     np.savetxt(f'data/train_data{train_data[i:i*1000].shape}_starting_at_22977.csv', train_data[i:i*1000].flatten(), delimiter=',')
    #     np.savetxt(f'data/train_values{train_values[i:i*1000].shape}_starting_at_22977.csv', train_values[i:i*1000].flatten(), delimiter=',')
    num_test_plays = 10
    print('reading data')
    train_data = np.genfromtxt('train_data(31007, 10, 10, 11).csv', delimiter=',').reshape((31007, 10, 10, 11))
    print('reading values')
    train_values = np.genfromtxt('train_values(31007, 199).csv', delimiter=',').reshape((31007, 199))
    model = create_model(train_data)
    print('Training model')
    start = datetime.datetime.now()
    model.fit(train_data[:-num_test_plays], train_values[:-num_test_plays], epochs=50)
    model.save(f'model')
    print(f'Finished in {datetime.datetime.now() - start}.\n')
    # first = True
    # for f in listdir('data'):
    #     if 'train_data' in f:
    #         if 'starting' in f:
    #             data = re.search(r'\(.*(?=\.)', f).group()
    #             value_file = f'data/train_values({data.split(", ")[0][1:]}, 199)_starting_at_22977.csv'
    #         else:
    #             data = re.search(r'\(.*\)', f).group()
    #             value_file = f'data/train_values({data.split(", ")[0][1:]}, 199).csv'
    #         data_file = f'data/train_data{data}.csv'
    #         print(f'data: {data}')
    #         print(f'data_file: {data_file}')
    #         print(f'value_file: {value_file}')
    #         train_data = np.genfromtxt(data_file, delimiter=',')
    #         train_data = train_data.reshape((int(len(train_data)/10/10/11), 10, 10, 11))
    #         train_values = np.genfromtxt(value_file, delimiter=',')
    #         train_values = train_values.reshape((int(len(train_values)/199), 199))
    #
    #         num_test_plays = 10
    #         if first:
    #             model = create_model(train_data)
    #             first = False
    #
    #         print('Training model')
    #         start = datetime.datetime.now()
    #         model.fit(train_data[:-num_test_plays], train_values[:-num_test_plays], epochs=50)
    #         model.save(f'model_after_data{data}')
    #         print(f'Finished in {datetime.datetime.now() - start}.\n')
    #
    print('Predicting')
    pred = model.predict(train_data[-num_test_plays:])
    print(f'Finished in {datetime.datetime.now() - start}.\n')

    with open('predictions.txt', 'w') as f:
        for p, a in zip(pred, train_values[-num_test_plays:]):
            for (p1, a1, i) in zip(p, a, range(len(a))):
                f.write('i: {: 3d}; Actual: {:f}; Predicted: {:f};\n'.format(i - 99, a1, p1))
            f.write('\n')
    print('predictions.txt wrote.')


def create_model(train_data):
    print('Compiling model')
    start = datetime.datetime.now()

    def crps_loss_func(y_true, y_pred):
        ret = tf.where(y_true >= 1, y_pred - 1, y_pred)
        ret = K.square(ret)
        per_play_loss = K.sum(ret, axis=1)
        total_loss = K.mean(per_play_loss)
        return total_loss

    def max_avg_pool_2D(x):
        return tf.keras.layers.MaxPooling2D(pool_size=(1, 10))(x) * .3 \
               + tf.keras.layers.AveragePooling2D(pool_size=(1, 10))(x) * .7

    def max_avg_pool_1D(x):
        # Possibly change 10 back to 11 if change data shape
        return tf.keras.layers.MaxPooling1D(pool_size=10)(x) * .3 \
               + tf.keras.layers.AveragePooling1D(pool_size=10)(x) * .7

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                               input_shape=train_data[0].shape))
    model.add(tf.keras.layers.Conv2D(160, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.Lambda(max_avg_pool_2D))
    model.add(tf.keras.layers.Lambda(lambda x: K.squeeze(x, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(160, kernel_size=1, strides=1, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(96, kernel_size=1, strides=1, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(96, kernel_size=1, strides=1, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Lambda(max_avg_pool_1D))
    model.add(tf.keras.layers.Lambda(lambda x: K.squeeze(x, 1)))
    model.add(tf.keras.layers.Dense(96, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(199, activation='softmax'))
    model.compile(optimizer='adam', loss=crps_loss_func, metrics=['accuracy'])

    print(f'Finished in {datetime.datetime.now() - start}.\n')
    return model


def get_test_train_data(data, df):
    print('Converting to training data')
    start = datetime.datetime.now()

    def convert_to_crps(yards):
        y = [0 for i in range(115)]
        for i in range(15 + yards, 115):
            y[i] = 1
        return np.pad(np.asarray(y, dtype='float'), (84, 0), constant_values=0)

    unstack = df.unstack(-1)
    num_plays = len(df.index.get_level_values(0).unique())
    train_data = np.reshape(unstack.to_numpy(), (num_plays, 10, 10, 11))
    # New dimensions: (num_plays, off, features, def)

    train_values = np.asarray(
        list(map(np.asarray, data.groupby(['GameId', 'PlayId']).first()["Yards"].apply(convert_to_crps).values)))

    print(f'Finished in {datetime.datetime.now() - start}.\n')
    return train_data, train_values


def create_features(data):
    print('Creating features for each play')
    start = datetime.datetime.now()

    off_labels = [f'Off{i}' for i in range(10)]
    def_labels = [f'Def{i}' for i in range(11)]
    groups = data.groupby(['GameId', 'PlayId']).groups
    index = pd.MultiIndex.from_product([groups, off_labels, def_labels], names=['Play', 'Off', 'Def'])
    df = pd.DataFrame(index=index,
                      columns=['DefSpeedX', 'DefSpeedY', 'DefRusherX', 'DefRusherY', 'DefRusherSpeedX',
                               'DefRusherSpeedY', 'OffDefX', 'OffDefY', 'OffDefSpeedX', 'OffDefSpeedY'])

    def get_def_speed(x, label):
        i = int(re.search(r'[0-9]+', x.name[2])[0])
        return data.loc[x.name[0][0], x.name[0][1]][f'DefenseS{label.lower()}'][i]

    def get_def_rusher(x, label):
        i = int(re.search(r'[0-9]+', x.name[2])[0])
        r = data.loc[x.name[0][0], x.name[0][1]][f'Rusher{label.upper()}']
        d = data.loc[x.name[0][0], x.name[0][1]][f'Defense{label.upper()}'][i]
        return d - r

    def get_def_rusher_speed(x, label):
        i = int(re.search(r'[0-9]+', x.name[2])[0])
        rusher_speed = data.loc[x.name[0][0], x.name[0][1]][f'RusherS{label.lower()}']
        def_speed = data.loc[x.name[0][0], x.name[0][1]][f'DefenseS{label.lower()}'][i]
        return def_speed - rusher_speed

    def get_off_def(x, label):
        def_i = int(re.search(r'[0-9]+', x.name[2])[0])
        off_i = int(re.search(r'[0-9]+', x.name[1])[0])
        d = data.loc[x.name[0][0], x.name[0][1]][f'Defense{label.upper()}'][def_i]
        o = data.loc[x.name[0][0], x.name[0][1]][f'Offense{label.upper()}'][off_i]
        return o - d

    def get_off_def_speed(x, label):
        def_i = int(re.search(r'[0-9]+', x.name[2])[0])
        off_i = int(re.search(r'[0-9]+', x.name[1])[0])
        off_speed = data.loc[x.name[0][0], x.name[0][1]][f'OffenseS{label.lower()}'][off_i]
        def_speed = data.loc[x.name[0][0], x.name[0][1]][f'DefenseS{label.lower()}'][def_i]
        return off_speed - def_speed

    s = datetime.datetime.now()
    df['DefSpeedX'] = df.apply(lambda x: get_def_speed(x, 'x'), axis=1)
    print(f'DefSpeedX done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['DefSpeedY'] = df.apply(lambda x: get_def_speed(x, 'y'), axis=1)
    print(f'DefSpeedY done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['DefRusherX'] = df.apply(lambda x: get_def_rusher(x, 'x'), axis=1)
    print(f'DefRusherX done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['DefRusherY'] = df.apply(lambda x: get_def_rusher(x, 'y'), axis=1)
    print(f'DefRusherY done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['DefRusherSpeedX'] = df.apply(lambda x: get_def_rusher_speed(x, 'x'), axis=1)
    print(f'DefRusherSpeedX done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['DefRusherSpeedY'] = df.apply(lambda x: get_def_rusher_speed(x, 'y'), axis=1)
    print(f'DefRusherSpeedY done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['OffDefX'] = df.apply(lambda x: get_off_def(x, 'x'), axis=1)
    print(f'OffDefX done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['OffDefY'] = df.apply(lambda x: get_off_def(x, 'y'), axis=1)
    print(f'OffDefY done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['OffDefSpeedX'] = df.apply(lambda x: get_off_def_speed(x, 'x'), axis=1)
    print(f'OffDefSpeedX done {datetime.datetime.now() - s}.')
    s = datetime.datetime.now()
    df['OffDefSpeedY'] = df.apply(lambda x: get_def_speed(x, 'y'), axis=1)
    print(f'OffDefSpeedY done {datetime.datetime.now() - s}.')

    print(f'Finished in {datetime.datetime.now() - start}.\n')
    return df


def combine_by_team(data):
    print('Combining data by team and split into x,y components.')
    start = datetime.datetime.now()
    data = data.set_index(["GameId", "PlayId"])

    offense = data['IsOffense'] & ~data['IsRusher']
    rusher = data['IsRusher']
    defense = ~data['IsOffense']

    data['OffenseX'] = data.loc[offense].groupby(['GameId', 'PlayId'])['X_std'].apply(list)
    data['OffenseY'] = data.loc[offense].groupby(['GameId', 'PlayId'])['Y_std'].apply(list)
    data['OffenseSx'] = data.loc[offense].groupby(['GameId', 'PlayId'])['S_x'].apply(list)
    data['OffenseSy'] = data.loc[offense].groupby(['GameId', 'PlayId'])['S_y'].apply(list)
    data['RusherX'] = data.loc[rusher].groupby(['GameId', 'PlayId'])['X_std'].first()
    data['RusherY'] = data.loc[rusher].groupby(['GameId', 'PlayId'])['Y_std'].first()
    data['RusherSx'] = data.loc[rusher].groupby(['GameId', 'PlayId'])['S_x'].first()
    data['RusherSy'] = data.loc[rusher].groupby(['GameId', 'PlayId'])['S_y'].first()
    data['DefenseX'] = data.loc[defense].groupby(['GameId', 'PlayId'])['X_std'].apply(list)
    data['DefenseY'] = data.loc[defense].groupby(['GameId', 'PlayId'])['Y_std'].apply(list)
    data['DefenseSx'] = data.loc[defense].groupby(['GameId', 'PlayId'])['S_x'].apply(list)
    data['DefenseSy'] = data.loc[defense].groupby(['GameId', 'PlayId'])['S_y'].apply(list)

    data = data.reset_index().drop_duplicates(subset=['GameId', 'PlayId'], keep='first')
    data = data.set_index(["GameId", "PlayId"])

    print(f'Finished in {datetime.datetime.now() - start}.\n')
    return data


def standardized_play_direction(data):
    print('Standardizing play direction')
    start = datetime.datetime.now()

    def get_is_rusher(x): return x['NflId'] == x['NflIdRusher']

    def get_is_offense(x): return (x['Team'] == 'home') == (x['PossessionTeam'] == x['HomeTeamAbbr'])

    def get_to_left(x): return x["PlayDirection"] == "left"

    def get_x_std(x): return 120 - x["X"] - 10 if x["ToLeft"] else x["X"] - 10

    def get_y_std(x): return 160 / 3 - x["Y"] if x["ToLeft"] else x["Y"]

    def get_dir_std_1(x): return x["Dir"] + 360 if x["ToLeft"] and x["Dir"] < 90 else x["Dir"]

    def get_dir_std_1_next(x): return x["Dir"] - 360 if (not x["ToLeft"]) and x["Dir"] > 270 else x["Dir_std_1"]

    def get_dir_std_2(x): return x["Dir_std_1"] - 180 if x["ToLeft"] else x["Dir_std_1"]

    def get_x_std_end(x): return x["S"] * math.cos((90 - x["Dir_std_2"]) * math.pi / 180) + x["X_std"]

    def get_y_std_end(x): return x["S"] * math.sin((90 - x["Dir_std_2"]) * math.pi / 180) + x["Y_std"]

    def get_s_x(x): return x["X_std_end"] - x["X_std"]

    def get_s_y(x): return x["Y_std_end"] - x["Y_std"]

    data["IsRusher"] = data.apply(get_is_rusher, axis=1)
    data["IsOffense"] = data.apply(get_is_offense, axis=1)
    data["ToLeft"] = data.apply(get_to_left, axis=1)
    data["X_std"] = data.apply(get_x_std, axis=1)
    data["Y_std"] = data.apply(get_y_std, axis=1)
    data["Dir_std_1"] = data.apply(get_dir_std_1, axis=1)
    data["Dir_std_1"] = data.apply(get_dir_std_1_next, axis=1)
    data["Dir_std_2"] = data.apply(get_dir_std_2, axis=1)
    data["X_std_end"] = data.apply(get_x_std_end, axis=1)
    data["Y_std_end"] = data.apply(get_y_std_end, axis=1)
    data["S_x"] = data.apply(get_s_x, axis=1)
    data["S_y"] = data.apply(get_s_y, axis=1)

    print(f'Finished in {datetime.datetime.now() - start}.\n')
    return data


def get_data():
    print('Reading csv.')
    start = datetime.datetime.now()
    data = pd.read_csv('train.csv')
    keep_cols = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'Dis', 'Orientation', 'Dir', 'NflId', 'PossessionTeam',
                 'NflIdRusher', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Yards', 'PlayDirection']
    data.drop([x for x in data.columns if x not in keep_cols], axis=1, inplace=True)
    print(f'Finished in {datetime.datetime.now() - start}.\n')
    return data


main()
