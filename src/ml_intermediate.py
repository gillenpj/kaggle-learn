# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../data/train.csv', index_col='Id')
X_test_full = pd.read_csv('../data/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# %%
X_train.head()

# %%
from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# %%
from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))

# %%
my_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=1)

# %%
# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

# %% [markdown]
# # Complete Upgraded Code (ready to run)

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ======================
#  LOAD DATA
# ======================
X_full = pd.read_csv('../data/train.csv', index_col='Id')
X_test_full = pd.read_csv('../data/test.csv', index_col='Id')

y = X_full.SalePrice
X_full = X_full.drop(['SalePrice'], axis=1)

# ======================
#  FEATURE ENGINEERING
# ======================

def add_engineered_features(df):
    df = df.copy()
    # Ages
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Total bathrooms
    df["TotalBaths"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Total square footage indicators
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"]
        + df["3SsnPorch"] + df["ScreenPorch"]
    )

    # Binary indicators
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)

    return df

X_full = add_engineered_features(X_full)
X_test_full = add_engineered_features(X_test_full)

# ======================
# SELECT FEATURES
# ======================

selected_features = [
    # Numeric — strong predictors
    "OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "GarageCars", "GarageArea",
    "LotArea", "YearBuilt", "YearRemodAdd",

    # Engineered
    "HouseAge", "RemodAge", "TotalBaths",
    "TotalSF", "TotalPorchSF", "HasPool", "HasGarage",
    "HasFireplace", "HasBsmt",

    # Categorical — strong signals
    "Neighborhood", "MSZoning", "Exterior1st", "Exterior2nd",
    "Foundation", "KitchenQual", "HeatingQC",
    "GarageFinish", "FireplaceQu", "BsmtQual",
    "BsmtCond", "BsmtExposure"
]

X = X_full[selected_features].copy()
X_test = X_test_full[selected_features].copy()

# One-hot encode
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Align columns
X, X_test = X.align(X_test, join='left', axis=1)
X_test = X_test.fillna(0)

# ======================
# TRAIN/VALIDATION SPLIT
# ======================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# ======================
# RANDOM FOREST MODEL
# ======================
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="sqrt",
    criterion="absolute_error",
    random_state=0,
    n_jobs=-1,
)

model.fit(X_train, y_train)
preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)
print("Validation MAE:", mae)

# ======================
# FIT ON FULL DATA & EXPORT
# ======================
model.fit(X, y)
preds_test = model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission_feat.csv', index=False)
print("Submission file created!")

# %% [markdown]
# # Incremental XGBoost code

# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# ======================
# SIMPLE XGBOOST MODEL
# ======================

xgb_model = XGBRegressor(
    n_estimators=500,       # number of trees
    learning_rate=0.05,    # lower LR, more trees
    max_depth=4,           # shallow trees → less overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=0,
    n_jobs=-1,
)

# Fit on training split
xgb_model.fit(X_train, y_train)

# Validate
xgb_valid_preds = xgb_model.predict(X_valid)
xgb_mae = mean_absolute_error(y_valid, xgb_valid_preds)
print(f"XGBoost Validation MAE: {xgb_mae:,.3f}")

# ======================
# TRAIN ON FULL DATA & CREATE SUBMISSION
# ======================

final_xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=0,
    n_jobs=-1,
)

final_xgb_model.fit(X, y)

xgb_test_preds = final_xgb_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': xgb_test_preds})
output.to_csv('submission_xgb.csv', index=False)
print("Saved submission_xgb.csv")

# %% [markdown]
# # Apply log transform to SalePrice and drop in a LightGBM model

# %%
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# Log-transform target only
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)
y_log = np.log1p(y)

# Simple, conservative LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=0,
)

lgbm_model.fit(X_train, y_train_log)

valid_preds_log = lgbm_model.predict(X_valid)
valid_preds = np.expm1(valid_preds_log)
print("LGBM MAE:", mean_absolute_error(y_valid, valid_preds))

# Train on full data if you like the validation MAE
final_lgbm = LGBMRegressor(
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=0,
)

final_lgbm.fit(X, y_log)
test_preds_log = final_lgbm.predict(X_test)
test_preds = np.expm1(test_preds_log)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': test_preds})
output.to_csv('submission_lgbm.csv', index=False)

# %%
