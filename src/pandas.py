# %%
import pandas as pd
pd.set_option('display.max_rows', 5)

# %%
pd.DataFrame({'Apples': [35, 34], 'Bananas': [21, 34]}, index=['2017 Sales', '2018 Sales'])

# %%
pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')

# %%
reviews = pd.read_csv('../data/winemag-data-130k-v2.csv', index_col=0)

# %%
reviews.head()

# %%
reviews.country

# %%
reviews['country']

# %%
reviews.iloc[0]

# %%
reviews.iloc[:, 0]

# %%
reviews.iloc[:3, 0]

# %%
reviews.iloc[-5, :]

# %%
reviews.index

# %%
reviews.set_index('title')

# %%
reviews.loc[reviews.country == 'Italy']

# %%
reviews.loc[reviews.country.isin(['Italy', 'France'])]

# %%
desc = reviews.description

# %%
first_description = reviews.loc[0, 'description']

# %%
first_row = reviews.loc[0, :]

# %%
first_descriptions = pd.Series(reviews.description[0:10], name = 'description')

# %%
first_descriptions

# %%
sample_reviews = reviews.iloc[[1, 2, 3, 5, 8], :]

# %%
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]

# %%
df

# %%
df = reviews.loc[0:100, ['country', 'variety']]

# %%
italian_wines = reviews.loc[reviews.country == 'Italy', :]

# %%
reviews.columns

# %%
reviews.columns

# %%
top_oceania_wines = reviews.loc[reviews.country.isin(['New Zealand', 'Australia']) & (reviews.points >= 95)]

# %%
reviews.columns

# %%
reviews.points.mean()

# %%
reviews.country.unique()

# %%
reviews.country.value_counts()

# %%
centred_price = reviews.price - reviews.price.mean()

# %%
points_to_price = reviews.points / reviews.price

# %%
points_to_price.max()

# %%
reviews.loc[points_to_price == points_to_price.max(), 'title']

# %%
descriptor_counts = [reviews.description.str.contains('tropical').sum(), reviews.description.str.contains('fruit').sum()]

# %%
descriptor_counts

# %%
def star_rating(row):
    if row.country == 'Canada' or row.points >= 95:
        return '***'
    elif row.points >= 85:
        return '**'
    else:
        return '*'

reviews.groupby(reviews.apply(star_rating, axis=1))['price'].mean().rename('mean_price')

# %% [markdown]
# # Applying a function to each row (axis = 1)

# %%
def points_per_price(row):
    return row['points'] / row['price']

reviews['value'] = reviews.apply(points_per_price, axis=1)

# %%
reviews

# %% [markdown]
# # Apply a function to each column (axis=0)

# %%
pd.set_option('display.max_rows', None)       # show all rows in any output
pd.set_option('display.max_columns', None)    # show all columns if needed

reviews.apply(lambda col: col.isna().sum(), axis=0)

# %%
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# %%
print(reviews.apply(lambda col: col.isna().sum(), axis=0))

# %% [markdown]
# # Grouping and Sorting

# %%
reviews.taster_twitter_handle.value_counts()

# %%
reviews.groupby('price').points.max()

# %%
reviews.variety

# %%
reviews.groupby('variety').price.agg(['max', 'min']).sort_values(by='max', ascending=False)

# %%
reviewer_mean_ratings = reviews.groupby('taster_name').price.mean()

# %%
reviewer_mean_ratings.describe()

# %%
print(reviews.groupby(['country', 'variety']).description.agg(['count']).sort_values(by='count', ascending=False))

# %% [markdown]
# # Data Types and Missing Values

# %%
reviews.price.dtype

# %%
reviews.dtypes

# %%
reviews.index.dtype

# %%
reviews.points.dtype

# %%
reviews.points.astype(str)

# %%
n_missing_values = reviews.price.isna().sum()

# %%
print(n_missing_values)

# %%
reviews['region_1'] = reviews.region_1.map(lambda x: 'Unknown' if pd.isna(x) else x)

# %%
reviews['region_1'].value_counts()

# %% [markdown]
# # Renaming and Combining

# %%
reviews2 = reviews.rename({'region_1': 'region', 'region_2': 'locale'})

# %%
reviews2.index.name = 'wines'

# %%
# concatenation
# join, on common index/indices
