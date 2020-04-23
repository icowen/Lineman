import json

import pandas as pd

data = pd.read_csv('netdata_with_ball.csv', dtype='float32',
                   converters={'PassResult': lambda x: 'R' if pd.isna(x) else x,
                               'pass.frame': lambda x: 0 if x == '' else int(x),
                               'playId': lambda x: int(x),
                               'gameId': lambda x: int(x),
                               'frame.id': lambda x: int(x),
                               'sack.ind': lambda x: int(x),
                               'num_vec': lambda x: int(x)})
data.dropna(inplace=True)
data.drop(['X'], axis=1, inplace=True)
print(data.dtypes.to_string())
# out_data = list(data.loc[data['sack.ind'] == 1].groupby(['gameId', 'playId']).groups.keys())

# with open('games_with_sacks.json', 'w') as json_file:
#     json.dump(out_data, json_file)
