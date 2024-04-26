import pandas as pd
import numpy as np

SESSIONS_LOC = './raw_data/sessions.csv'
df_sessions = pd.read_csv(SESSIONS_LOC)

# Let's remove the duplicates from sessions
df_sessions = df_sessions.drop_duplicates()
# print(df_sessions.duplicated().value_counts()) # should only have 'False' now

# Let's do some EDA on the sessions
# print(df_sessions.columns)
# print(df_sessions.isnull().sum())
# print(df_sessions.isna().sum())
# print(df_sessions.describe())

# returns the number of unique devices used for a specific user id
# perhaps the number of different devices will correspond?
# more devices -> more wealth -> more likely to vacation
from functools import reduce
def all_devices_per_id(series):
    return len(reduce(lambda x, y: x if y in x else x + ';' + y, series).split(';'))

def reduce_actions(series):
    return reduce(lambda x, y: x + ';' + y, series)

def reduce_actions_len(series):
    return len(reduce_actions(series).split(';'))

# the same as all devices per id
def unique_actions(series):
    return len(reduce(lambda x, y: x if y in x else x + ';' + y, series).split(';'))

# Let's get some aggregate statistics about the sessions
df_sessions_total_seconds = df_sessions.groupby(['user_id'])['secs_elapsed'].agg(['count', 'sum'])
df_sessions_total_seconds['avg_time_per_session'] = df_sessions_total_seconds['sum'] / df_sessions_total_seconds['count']
df_sts_numeric_cols = df_sessions_total_seconds.select_dtypes(include=[np.number]).columns
# print(df_sessions_total_seconds[df_sts_numeric_cols].apply(zscore))

df_sessions_no_nulls_action = df_sessions.dropna(subset=['action'])
df_sessions_no_nulls_action_type = df_sessions.dropna(subset=['action_type'])

df_unique_devices = df_sessions.groupby('user_id')['device_type'].apply(all_devices_per_id)

df_actions = df_sessions_no_nulls_action.groupby('user_id').agg({'action': [reduce_actions, unique_actions, reduce_actions_len]})
df_actions.columns = df_actions.columns.droplevel()

df_action_types = df_sessions_no_nulls_action_type.groupby('user_id').agg({'action_type': [reduce_actions, unique_actions, reduce_actions_len]})
df_action_types.columns = df_action_types.columns.droplevel()

# has to be x IN action NOT x == action because ending action\n != action
def len_of_action_from_series(it, action):
    vals = it.split(';')
    f = filter(lambda x: x in action, vals)
    # print(list(f))
    return len(list(f))

# df_actions.reduce_actions.apply(len_of_action_from_series, args=("show",)).sum()
action_frames = []
actions_set = set(df_sessions_no_nulls_action.action.values)
for action in actions_set:
    action_frames.append(df_actions.reduce_actions.apply(len_of_action_from_series, args=(action,)))


action_frames.insert(0, df_actions)                    
action_result = pd.concat(action_frames, axis=1)

action_type_frames = []
action_type_set = set(df_sessions_no_nulls_action_type.action_type.values)
for action_type in action_type_set:
    action_type_frames.append(df_action_types.reduce_actions.apply(len_of_action_from_series, args=(action_type,)))


action_type_frames.insert(0, df_action_types)
action_type_result = pd.concat(action_type_frames, axis=1)
