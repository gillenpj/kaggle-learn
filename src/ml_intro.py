# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
# save filepath to variable for easier access
iowa_file_path = '../data/train.csv'
# read the data and store data in DataFrame called melbourne_data
iowa_data = pd.read_csv(iowa_file_path)
# print a summary of the data in Melbourne data
iowa_data.describe()

# %%
iowa_data.columns

# %%
y = iowa_data.SalePrice

# %%
iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_data[iowa_features]

# %%
X.describe()

# %%
X.head(5)

# %%
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
iowa_model = DecisionTreeRegressor(random_state=1)

# %% [markdown]
# ## First we'll fit a model without splitting out training and validation sets

# %%
# Fit model
iowa_model.fit(X, y)

# %%
print("Making predictions for the following 5 house:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

# %%
y.head()

# %%
# We get ridiculously good scores
from sklearn.metrics import mean_absolute_error

predicted_home_prices = iowa_model.predict(X)
print(f"{mean_absolute_error(y, predicted_home_prices):.1f}")

# %% [markdown]
# ## Now we'll split out training and validation data sets

# %%
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

# %%
# The score isn't quite so impressive!
val_predictions = iowa_model.predict(val_X)
print(f"{mean_absolute_error(val_y, val_predictions):.1f}")


# %% [markdown]
# ## Next up - Underfitting & Overfitting

# %%
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# %%
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# compare MAE with differing values of max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
for max_leaf_node in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_node, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" % (max_leaf_node, my_mae))

# %% [markdown]
# ## Fit Model Using All Data

# %%
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
scores

# %%
best_tree_size = min(scores, key=scores.get)
best_tree_size

# %%
# Fit the model with best_tree_size. Fill in argument to make optimal size.
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model
final_model.fit(train_X, train_y)

# %%
predicted_home_prices = final_model.predict(val_X)
print(f"{mean_absolute_error(val_y, predicted_home_prices):.1f}")

# %% [markdown]
# ## Using a Random Forest

# %%
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random state to 1.
rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_pred_vals = rf_model.predict(val_X)
print("Validate MAE for Random Forest Model: {}".format(mean_absolute_error(val_y, rf_pred_vals)))
