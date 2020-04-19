import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

matplotlib.use("TkAgg")

fig, ax = plt.subplots()
fig.set_tight_layout(True)
plt.xticks([], [])
plt.yticks([], [])
def_label = mpatches.Patch(color='red', label='Defense')
off_label = mpatches.Patch(color='blue', label='Offense')
plt.legend(handles=[def_label, off_label])

df = pd.read_csv('netdata.csv')
df.drop('X', inplace=True, axis=1)
df.fillna('Run', inplace=True)
gameId = 2017090700
playId = 68
play = df.loc[(df["playId"] == playId) & (df["gameId"] == gameId)]
plt.title(f'Play {playId}')

field_length = 120
field_width = 160 / 3
ax.set_xlim([0, field_length])
ax.set_ylim([0, field_width])

ax.set_facecolor('green')
ax.fill_between([0, 10], field_width, color='maroon')
ax.fill_between([110, field_length], field_width, color='gold')
scat = ax.scatter([], [], s=16, edgecolors='black')

for i in range(1, 12):
    ax.axvline(i * 10, c='white', zorder=0)
    if i < 6:
        ax.annotate(i * 10, ((i + 1) * 10 + 1, 5), c='white')
    elif i < 10:
        ax.annotate(100 - i * 10, ((i + 1) * 10 + 1, 5), c='white')


def get_new_x_y(frame_df):
    # Find columns with coordinate data
    x_cols = [c for c in frame_df.columns if re.match('.*_x$|^X_', c)]
    y_cols = [c for c in frame_df.columns if re.match('.*_y$|^Y_', c)]

    # Get coordinates for each player in the frame
    x = [frame_df[x].values[0] for x in x_cols]
    y = [frame_df[y].values[0] for y in y_cols]

    # Identify a team for each pair of coordinates
    labels = ['OL'] * 5 + ['def'] * 5 + ['off'] * 6 + ['def'] * 11

    # Return coordinate/team data frame
    return pd.DataFrame(dict(x=x, y=y, label=labels))


def update(frame):
    # Get data for this frame
    frame_df = play[play["frame.id"] == frame]

    # Clean coordinates
    data = get_new_x_y(frame_df)

    # Convert coordinate data frame to list of tuples and plot
    x_y = [(x, y) for x, y in zip(data['x'], data['y'])]
    scat.set_offsets(x_y)
    scat.set_edgecolors('black')

    # Identify offense vs defense
    colors = ['r' if x == 'def' else 'b' for x in data['label']]
    scat.set_color(colors)

    # Update frame label
    ax.set_xlabel(f'Frame {frame}')


if __name__ == '__main__':
    anim = FuncAnimation(fig, update, frames=play["frame.id"], interval=100)
    anim.save(f'play{playId}.gif', dpi=80, writer='imagemagick')
    # plt.show()
