import pandas as pd
COUNTRIES_LOC = './raw_data/countries.csv'
df_countries = pd.read_csv(COUNTRIES_LOC)

# We looked at sessions earlier, lets look at countries
# print(df_countries.shape)
# print(df_countries.describe())
# print(df_countries.head()) # might be something with lat long related to booking date? close-to-equator in winter vs far-from-equator in summer, also take into account southern vs northern hemisphere
# print(df_countries.columns)
# print(df_train.columns)


# when summer in that hemisphere starts
def summer_start(x):
    if x < 0:
        return 12
    return 6

# when summer in that hemisphere ends
# month 2 = february for southern hemisphere
def summer_end(x):
    if x < 0:
        return 2
    return 8

# Let's do countries a bit
df_countries['summer_start'] = df_countries['lat_destination'].apply(summer_start)
df_countries['summer_end'] = df_countries['lat_destination'].apply(summer_end)
df_countries.to_csv('countries_agg_data.csv')
