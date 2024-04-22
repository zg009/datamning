
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def category_encode(features: pd.DataFrame):

    cat_classes = features.select_dtypes(include=[object])

    enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    enc.fit(cat_classes)

    encoded_cat_data = enc.fit_transform(cat_classes)
    new_df = pd.DataFrame(encoded_cat_data, )

    return new_df


AGE_GENDER_BKTS_LOC = '/raw_data/age_gender_bkts.csv'
COUNTRIES_LOC = '/raw_data/countries.csv'
SESSIONS_LOC = '/raw_data/sessions.csv'
TEST_USERS_LOC = '/raw_data/test_users.csv'
TRAIN_USERS_LOC = '/raw_data/train_users_2.csv'

df_age_gender = pd.read_csv(AGE_GENDER_BKTS_LOC)
df_countries = pd.read_csv(COUNTRIES_LOC)
df_sessions = pd.read_csv(SESSIONS_LOC)
df_test = pd.read_csv(TEST_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])
df_train = pd.read_csv(TRAIN_USERS_LOC, parse_dates=['date_account_created', 'timestamp_first_active', 'date_first_booking'])

# df_sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()