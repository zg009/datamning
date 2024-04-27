import pandas as pd
import numpy as np

AGE_GENDER_BKTS_LOC = './raw_data/age_gender_bkts.csv'
df_age_gender = pd.read_csv(AGE_GENDER_BKTS_LOC)

df_age_gender['population_in_thousands'] = df_age_gender.population_in_thousands.astype('Int32')

df_age_gender['join_key'] = (df_age_gender['age_bucket'] + '_' + df_age_gender['country_destination']).astype('object')
df_age_gender_male = df_age_gender.loc[df_age_gender['gender'] == 'male']
df_age_gender_male = df_age_gender.drop(columns=['country_destination', 'age_bucket'])
df_age_gender_female = df_age_gender.loc[df_age_gender['gender'] == 'female']
df_age_gender_joined = df_age_gender_female.merge(df_age_gender_male, left_on='join_key', right_on='join_key')
df_age_gender_joined = df_age_gender_joined.loc[df_age_gender_joined['gender_y'] == 'male']
df_age_gender_joined = df_age_gender_joined.drop(columns=['year_x', 'join_key', 'year_y'])
df_age_gender_joined['male_pop_diff'] = df_age_gender_joined['population_in_thousands_y'] - df_age_gender_joined['population_in_thousands_x']
df_age_gender_joined['more_male'] = np.where(df_age_gender_joined['male_pop_diff'] > 0, 1, 0)
df_age_gender_joined.to_csv('age_gender_agg_data.csv')
