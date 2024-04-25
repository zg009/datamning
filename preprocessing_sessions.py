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

df_sessions_no_nulls_action = df_sessions.dropna(subset=['action', 'action_type'])
df_unique_devices = df_sessions.groupby('user_id').agg({'device_type': all_devices_per_id}).reset_index()
# something wrong when building this
df_actions = df_sessions_no_nulls_action.groupby('user_id').agg({'action': [reduce_actions, unique_actions, reduce_actions_len]})
df_action_types = df_sessions_no_nulls_action.groupby('user_id').agg({'action_type': [reduce_actions, unique_actions, reduce_actions_len]})
df_actions.columns = df_actions.columns.droplevel()

def len_of_action_from_series(it, action):
    f = filter(lambda x: x == action, it.split(';'))
    return len(list(f))
