import pandas as pd
from sklearn.preprocessing import OneHotEncoder
TEST_USERS_LOC = './raw_data/test_users.csv'
df_test = pd.read_csv(TEST_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])
TRAIN_USERS_LOC = './raw_data/train_users_2.csv'
df_train = pd.read_csv(TRAIN_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])

# Let's inspect the test data for inconsistencies
# print(df_train['age'].value_counts()) # stuff is fucked up here
# print(df_train['country_destination'].value_counts())
# print(df_train['language'].value_counts())
# print(df_train['gender'].value_counts()) # this means have to encode not with greater but greater, less, or indifferent

df_train['date_account_created_year'] = df_train.date_account_created.dt.year
df_train['date_account_created_month'] = df_train.date_account_created.dt.month
df_train['date_account_created_day'] = df_train.date_account_created.dt.day

# df_train['date_first_booking_year'] = df_train.date_first_booking.dt.year.astype('Int32')
# df_train['date_first_booking_month'] = df_train.date_first_booking.dt.month.astype('Int32')
# df_train['date_first_booking_day'] = df_train.date_first_booking.dt.day.astype('Int32')

df_train['date_timestamp_first_active_year'] = df_train.timestamp_first_active.dt.year.astype('Int32')
df_train['date_timestamp_first_active_month'] = df_train.timestamp_first_active.dt.month.astype('Int32')
df_train['date_timestamp_first_active_day'] = df_train.timestamp_first_active.dt.day.astype('Int32')
df_train.drop(['timestamp_first_active', 'date_first_booking', 'date_account_created'], axis=1, inplace=True)

df_train['age'] = df_train.age.astype('Int32')
df_train['gender'] = df_train.gender.str.lower()

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

def category_encode(features: pd.DataFrame):
    cat_classes = features.select_dtypes(include=[object])
    enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    enc.fit(cat_classes)
    encoded_cat_data = enc.fit_transform(cat_classes)
    new_df = pd.DataFrame(encoded_cat_data, )
    return new_df


def encode_age(num):
    if num > 1900:
        return int(2015 - num)
    return num


def encode_age_range(num, a):
    if num < 16 or num > 120:
        return int(a)
    return num


df_train['age'] = df_train.age.apply(encode_age)
df_train['age'] = df_train.age.apply(encode_age_range, args=(df_train.age.mean(),))
df_train['age'] = df_train.age.fillna(int(df_train.age.mean()))
df_train['age_bucket'] = df_train.age.apply(lambda x: next((v for k, v in age_buckets_mapping.items() if x in k), 0))
df_train['first_affiliate_tracked'] = df_train.first_affiliate_tracked.fillna('untracked')

# One hot encode some columns
# categorical columns
categorical_columns = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'signup_app', 'first_device_type', 'first_browser', 'age_bucket']
cat_classes = df_train[categorical_columns]
enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
enc.fit(cat_classes)
encoded_cat_data = enc.fit_transform(cat_classes)
df_train_encoded = pd.DataFrame(encoded_cat_data,)
df_train_ohe = pd.concat([df_train, df_train_encoded], axis=1)
df_train_ohe = df_train_ohe.drop(categorical_columns, axis=1)
# df_train_ohe.to_csv('train_ohe.csv')

# load in other frames
df_sessions = pd.read_csv('sessions_agg_data.csv')
# df_countries = pd.read_csv('countries_agg_data.csv')
# df_age_gender = pd.read_csv('age_gender_agg_data.csv')
# df_age_gender['age_gender_key'] = df_age_gender['age_bucket'] + '_' + df_age_gender['country_destination']

