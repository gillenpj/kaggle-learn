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
top_oceania_wines = reviews.loc[reviews.country.isin(['New Zealand', 'Australia']) & (reviews.points >= 95)]

# %%
top_oceania_wines

# %%
