import pandas as pd
from sklearn.preprocessing import OneHotEncoder

TEST_USERS_LOC = './raw_data/test_users.csv'
TRAIN_USERS_LOC = './raw_data/train_users_2.csv'


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


def booked(v):
    if 'NDF' in v:
        return False
    return True

### Function that does preprocessing for both train and test datasets
# type = test or train
def preprocessing_airbnb(df, type):
    df_copy = df

    ### Split date of account created into three seperate columns for training set
    df_copy['date_account_created_year'] = df_copy.date_account_created.dt.year
    df_copy['date_account_created_month'] = df_copy.date_account_created.dt.month
    df_copy['date_account_created_day'] = df_copy.date_account_created.dt.day

    ### Split date timestamp of first active into three seperate columns
    df_copy['date_timestamp_first_active_year'] = df_copy.timestamp_first_active.dt.year.astype('Int32')
    df_copy['date_timestamp_first_active_month'] = df_copy.timestamp_first_active.dt.month.astype('Int32')
    df_copy['date_timestamp_first_active_day'] = df_copy.timestamp_first_active.dt.day.astype('Int32')

    ### Drop original variables that were split
    df_copy.drop(['timestamp_first_active', 'date_first_booking', 'date_account_created'], axis=1, inplace=True)

    ### Age to int and gender to all lowercase
    df_copy['age'] = df_copy.age.astype('Int32')
    df_copy['gender'] = df_copy.gender.str.lower()

    ### Subtract age where weird people put in their birth year
    df_copy['age'] = df_copy.age.apply(encode_age)

    ### Encode age if beyond weird age rage (below 16 or above 120)
    df_copy['age'] = df_copy.age.apply(encode_age_range, args=(df_train.age.mean(),))

    ### Give mean to ages where NA
    df_copy['age'] = df_copy.age.fillna(int(df_copy.age.mean()))

    df_copy['age_bucket'] = df_copy.age.apply(lambda x: next((v for k, v in age_buckets_mapping.items() if x in k), 0))
    df_copy['first_affiliate_tracked'] = df_copy.first_affiliate_tracked.fillna('untracked')
    
    if type == 'train':
        #Give boolean if books or NDF
        df_copy['booked'] = df_copy.country_destination.apply(booked)

    ### One Hot Encoder on category classes
    categorical_columns = ['gender', 'first_affiliate_tracked', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'signup_app', 'first_device_type', 'first_browser', 'age_bucket']
    cat_classes = df_copy[categorical_columns]
    enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    enc.fit(cat_classes)
    encoded_cat_data = enc.fit_transform(cat_classes)
    df_encoded = pd.DataFrame(encoded_cat_data,)

    # Combine new encoded columns with original train set
    df_comb = pd.concat([df_copy, df_encoded], axis=1)
    # Drop all categorial columns
    df_comb = df_comb.drop(categorical_columns, axis=1)

    ### load in other frames
    df_sessions = pd.read_csv('sessions_agg_data.csv')
    # df_countries = pd.read_csv('countries_agg_data.csv')
    # df_age_gender = pd.read_csv('age_gender_agg_data.csv')
    # df_age_gender['age_gender_key'] = df_age_gender['age_bucket'] + '_' + df_age_gender['country_destination']
    df_all = df_comb.merge(df_sessions, left_on='id', right_on='user_id')

    df_left = df_comb.merge(df_sessions, how='left', left_on='id', right_on='user_id')

    # this is just for me since i did not want to rerun the sessions generating code
    df_all['avg_time_per_session'] = df_all.avg_time_per_session.fillna(0)
    df_left['avg_time_per_session'] = df_left.avg_time_per_session.fillna(0)

    if type == 'train':
        # alright the preprocessing is finished
        df_all.to_csv('training_data.csv')
        # added these
        df_left = df_left.drop(['user_id'], axis=1)
        df_left = df_left.fillna(0.0)
        df_left.to_csv('sparse_training_data.csv')

    if type == 'test':
        # alright the preprocessing is finished
        df_all.to_csv('testing_data.csv')
        # added these
        df_left = df_left.drop(['user_id'], axis=1)
        df_left = df_left.fillna(0.0)
        df_left.to_csv('sparse_testing_data.csv')

    print(df_all.describe())
    
    return df_all

df_train = pd.read_csv(TRAIN_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])
df_test = pd.read_csv(TEST_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])

preprocessing_airbnb(df_train, 'train')
preprocessing_airbnb(df_test, 'test')