import pandas as pd

TRAINING_DATA = './training_data.csv'
df = pd.read_csv(TRAINING_DATA, index_col='id')
df = df.drop(df.columns[[0]], axis=1)
X = df.drop(['country_destination'], axis=1)
y = df.country_destination