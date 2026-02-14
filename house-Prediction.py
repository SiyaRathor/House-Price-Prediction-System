import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

# LOAD DATA
dataset = pd.read_csv('train.csv')

dataset.fillna({
    "LotFrontage": dataset["LotFrontage"].mean(),
    "MasVnrType": "None",
    "PoolQC": "No Pool",
    "MiscFeature": "No Feature",
    "Alley": "No Alley",
    "Fence": "No Fence",
    "FireplaceQu": "No Fireplace",
    "GarageType": "No Garage",
    "GarageQual": "No Garage",
    "GarageCond": "No Garage",
    "GarageFinish": "No Garage",
    "GarageYrBlt": 0,
    "BsmtExposure": "No Basement",
    "BsmtFinType1": "No Basement",
    "BsmtFinType2": "No Basement",
    "BsmtQual": "No Basement",
    "BsmtCond": "No Basement",
    "MasVnrArea": 0,
    "Electrical": "No Electrical",
}, inplace=True)


dataset.drop(['Id'], axis=1, inplace=True)

# ENCODING BEFORE SPLIT 
dataset = pd.get_dummies(dataset, drop_first=True)

# FEATURES & TARGET
X = dataset.drop('SalePrice', axis=1)
y = np.log1p(dataset['SalePrice'])

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SCALING ONLY FOR LINEAR MODEL
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
print("Linear Regression CV Score:", cv_scores.mean())  #average performance of moder at different  folds of data

# RANDOM FOREST (NO SCALING)
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("R2 Score (Random Forest):", r2_score(y_test, rf_pred))

# XGBOOST
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("R2 Score (XGBoost):", r2_score(y_test, xgb_pred))
print("MSE (XGBoost):", mean_squared_error(y_test, xgb_pred))

# VISUALIZATION
plt.figure(figsize=(6,6))
plt.scatter(y_test, xgb_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (XGBoost)")
plt.show()

# Residual Plot
residuals = y_test - xgb_pred

plt.figure(figsize=(6,4))
plt.scatter(xgb_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Feature Importance
importance = pd.Series(xgb.feature_importances_, index=X.columns)
importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()


joblib.dump(xgb, "house_price_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

