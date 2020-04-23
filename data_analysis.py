import json

import pandas as pd

data = pd.read_csv('netdata.csv', dtype='float32',
                   converters={'PassResult': lambda x: 'R' if pd.isna(x) else x})
data.dropna(inplace=True)
data.drop(['X'], axis=1, inplace=True)
out_data = list(data.loc[data['sack.ind'] == 1].groupby(['gameId', 'playId']).groups.keys())

with open('games_with_sacks.json', 'w') as json_file:
    json.dump(out_data, json_file)
