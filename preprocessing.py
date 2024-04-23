import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore

def category_encode(features: pd.DataFrame):

    cat_classes = features.select_dtypes(include=[object])

    enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    enc.fit(cat_classes)

    encoded_cat_data = enc.fit_transform(cat_classes)
    new_df = pd.DataFrame(encoded_cat_data, )

    return new_df


AGE_GENDER_BKTS_LOC = './raw_data/age_gender_bkts.csv'
COUNTRIES_LOC = './raw_data/countries.csv'
SESSIONS_LOC = './raw_data/sessions.csv'
TEST_USERS_LOC = './raw_data/test_users.csv'
TRAIN_USERS_LOC = './raw_data/train_users_2.csv'

df_age_gender = pd.read_csv(AGE_GENDER_BKTS_LOC)
df_countries = pd.read_csv(COUNTRIES_LOC)
df_sessions = pd.read_csv(SESSIONS_LOC)
df_test = pd.read_csv(TEST_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])
df_train = pd.read_csv(TRAIN_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])

# print(df_train.head())

# Let's check for duplication...
# print(df_age_gender.duplicated().value_counts())
# print(df_countries.duplicated().value_counts())
# print(df_sessions.duplicated().value_counts()) # has duplicates
# print(df_test.duplicated().value_counts())
# print(df_train.duplicated().value_counts())

# Let's remove the duplicates from sessions
df_sessions = df_sessions.drop_duplicates()
# print(df_sessions.duplicated().value_counts()) # should only have 'False' now

# Let's do some EDA on the sessions
# print(df_sessions.columns)
# print(df_sessions.isnull().sum())
# print(df_sessions.isna().sum())
# print(df_sessions.describe())

# Let's get some aggregate statistics about the sessions
# df_sessions_total_seconds = df_sessions.groupby(['user_id'])['secs_elapsed'].agg(['count', 'sum'])
# df_sessions_total_seconds['avg_time_per_session'] = df_sessions_total_seconds['sum'] / df_sessions_total_seconds['count']
# print(df_sessions_total_seconds.dtypes)
# print(df_sessions_total_seconds.describe())
# df_sts_numeric_cols = df_sessions_total_seconds.select_dtypes(include=[np.number]).columns
# print(df_sessions_total_seconds[df_sts_numeric_cols].apply(zscore))

# Let's inspect the test data for inconsistencies
# print(df_train['age'].value_counts()) # stuff is fucked up here
# print(df_train['country_destination'].value_counts())
# print(df_train['language'].value_counts())
# print(df_train['gender'].value_counts()) # this means have to encode not with greater but greater, less, or indifferent

# Let's check for nulls
# print(df_age_gender.isnull().sum()) # no nulls
# print(df_countries.isnull().sum()) # no nulls
# print(df_sessions.isnull().sum()) # come back to this later
# print(df_test.shape)
# print(df_test.isnull().sum()) # maybe just drop the columns which have nulls here?

# print(df_train.shape)
# print(df_train.isnull().sum()) # has nulls, different processing methods available

# We looked at sessions earlier, lets look at countries
# print(df_countries.shape)
# print(df_countries.describe())
# print(df_countries.head()) # might be something with lat long related to booking date? close-to-equator in winter vs far-from-equator in summer, also take into account southern vs northern hemisphere
# print(df_countries.columns)
# print(df_train.columns)

df_train['date_account_created_year'] = df_train.date_account_created.dt.year
df_train['date_account_created_month'] = df_train.date_account_created.dt.month
df_train['date_account_created_day'] = df_train.date_account_created.dt.day

df_train['date_first_booking_year'] = df_train.date_first_booking.dt.year.astype('Int32')
df_train['date_first_booking_month'] = df_train.date_first_booking.dt.month.astype('Int32')
df_train['date_first_booking_day'] = df_train.date_first_booking.dt.day.astype('Int32')

df_train['date_timestamp_first_active_year'] = df_train.timestamp_first_active.dt.year.astype('Int32')
df_train['date_timestamp_first_active_month'] = df_train.timestamp_first_active.dt.month.astype('Int32')
df_train['date_timestamp_first_active_day'] = df_train.timestamp_first_active.dt.day.astype('Int32')

df_train['age'] = df_train.age.astype('Int32')

age_buckets_mapping = {
    range(0, 5): "0-4",
    range(5, 10): "5-9",
    range(10, 15): "10-14",
    range(15, 20): "15-19",
    range(20, 25): "20-24",
    range(25, 30): "25-29",
    range(30, 35): "30-34",
    range(35, 40): "35-39",
    range(40, 45): "40-44",
    range(45, 50): "45-49",
    range(50, 55): "50-54",
    range(55, 60): "55-59",
    range(60, 65): "60-64",
    range(65, 70): "65-69",
    range(70, 75): "70-74",
    range(75, 80): "75-79",
    range(80, 85): "80-84",
    range(85, 90): "85-89",
    range(90, 95): "90-94",
    range(95, 100): "95-99",
    range(100, 130): "100+"
}

df_age_gender['population_in_thousands'] = df_age_gender.population_in_thousands.astype('Int32')

# maybe other solutions for age interpolation?
# just drop for now
df_train = df_train.dropna(subset=['age'])
df_train['age'] = df_train.age.astype('Int32')
df_train = df_train.query('age < 120') # probably should be over 16 and under 120
df_train['age_bucket'] = df_train.age.apply(lambda x: next((v for k, v in age_buckets_mapping.items() if x in k), 0))

df_age_gender['join_key'] = (df_age_gender['age_bucket'] + '_' + df_age_gender['country_destination']).astype('object')
df_age_gender_male = df_age_gender.loc[df_age_gender['gender'] == 'male']
df_age_gender_male = df_age_gender.drop(columns=['country_destination', 'age_bucket'])
df_age_gender_female = df_age_gender.loc[df_age_gender['gender'] == 'female']
df_age_gender_joined = df_age_gender_female.merge(df_age_gender_male, left_on='join_key', right_on='join_key')
df_age_gender_joined = df_age_gender_joined.loc[df_age_gender_joined['gender_y'] == 'male']
df_age_gender_joined = df_age_gender_joined.drop(columns=['key_x', 'join_key', 'key_y'])
df_age_gender_joined['male_pop_diff'] = df_age_gender_joined['population_in_thousands_y'] - df_age_gender_joined['population_in_thousands_x']
df_age_gender_joined['more_male'] = np.where(df_age_gender_joined['male_pop_diff'] > 0, 1, 0)
df_age_gender_joined.to_csv('age_gender_joined.csv')