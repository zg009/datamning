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

# Let's check for duplication...
# print(df_age_gender.duplicated().value_counts())
# print(df_countries.duplicated().value_counts())
# print(df_sessions.duplicated().value_counts()) # has duplicates
# print(df_test.duplicated().value_counts())
# print(df_train.duplicated().value_counts())

# Let's check for nulls
# print(df_age_gender.isnull().sum()) # no nulls
# print(df_countries.isnull().sum()) # no nulls
# print(df_sessions.isnull().sum()) # come back to this later
# print(df_test.shape)
# print(df_test.isnull().sum()) # maybe just drop the columns which have nulls here?
# print(df_train.shape)
# print(df_train.isnull().sum()) # has nulls, different processing methods available

