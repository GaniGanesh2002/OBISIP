# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# %%
df = pd.read_csv("Housing.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# %%
print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# %%
numeric_cols = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))
sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Matrix")
plt.show()

# %%
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

print("\nCategorical Columns:")
print(categorical_cols)

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nEncoded Shape:", df_encoded.shape)


# %%
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)


# %%
model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# %%
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("\nCross Validation R² Scores:")
print(cv_scores)

print("Average CV Score:", cv_scores.mean())


# %%
y_pred = model.predict(X_test)


# %%
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("="*40)

print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R² Score:", round(r2,4))

# %%
plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()

# %%
residuals = y_test - y_pred

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()

# %%
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients["Absolute_Impact"] = abs(
    coefficients["Coefficient"]
)

coefficients = coefficients.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\nTop Influential Features")
print(coefficients.head(10))


# %%
plt.figure(figsize=(10,6))

sns.barplot(
    data=coefficients.head(10),
    x='Coefficient',
    y='Feature'
)

plt.title("Top 10 Influential Features")
plt.show()

# %%
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = sm.add_constant(X)

X_vif = X_vif.astype(int, errors='ignore')

X_vif = X_vif.astype(float)

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns

vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print(vif_data.sort_values("VIF", ascending=False))

# %%
X_vif = sm.add_constant(X)

# %%
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nSample Predictions")
print(comparison.head(10))


# %%
print("\nIntercept:")
print(model.intercept_)

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:30s} : {coef:.2f}")


# %%
area = float(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
stories = int(input("Enter Stories: "))
parking = int(input("Enter Parking spaces: "))

input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking,
                            1, 0, 0, 0, 1, 1, 0, 0]], columns=X.columns)

predicted_price = model.predict(input_data)

print("\nPredicted House Price:", predicted_price[0])


