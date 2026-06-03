# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %%
df = pd.read_csv("WineQT.csv")

# %%
print(df.head())

# %%
print("\nDataset Shape:")
print(df.shape)


# %%
print("\nDataset Information:")
print(df.info())

# %%
print("\nMissing Values:")
print(df.isnull().sum())

# %%
print("\nStatistical Summary:")
print(df.describe())

# %%
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# %%
plt.figure(figsize=(6,4))
sns.barplot(x='quality', y='alcohol', data=df)
plt.title("Alcohol vs Quality")
plt.show()

# %%
plt.figure(figsize=(6,4))
sns.barplot(x='quality', y='volatile acidity', data=df)
plt.title("Volatile Acidity vs Quality")
plt.show()

# %%
df = df.drop('Id', axis=1)


# %%
df['quality'] = [1 if x >= 7 else 0 for x in df['quality']]

print(df['quality'].value_counts())


# %%
X = df.drop('quality', axis=1)
y = df['quality']


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST CLASSIFIER =====")

print("Accuracy Score:")
print(accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# %%
sgd_model = SGDClassifier(random_state=42)

sgd_model.fit(X_train, y_train)

sgd_pred = sgd_model.predict(X_test)

print("\n===== SGD CLASSIFIER =====")

print("Accuracy Score:")
print(accuracy_score(y_test, sgd_pred))

print("\nClassification Report:")
print(classification_report(y_test, sgd_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, sgd_pred))

# %%
svc_model = SVC()

svc_model.fit(X_train, y_train)

svc_pred = svc_model.predict(X_test)

print("\n===== SUPPORT VECTOR CLASSIFIER =====")

print("Accuracy Score:")
print(accuracy_score(y_test, svc_pred))

print("\nClassification Report:")
print(classification_report(y_test, svc_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svc_pred))

# %%
rf_acc = accuracy_score(y_test, rf_pred)
sgd_acc = accuracy_score(y_test, sgd_pred)
svc_acc = accuracy_score(y_test, svc_pred)

models = ['Random Forest', 'SGD', 'SVC']
accuracy = [rf_acc, sgd_acc, svc_acc]

plt.figure(figsize=(8,5))
sns.barplot(x=models, y=accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0,1)

plt.show()

# %%
importance = rf_model.feature_importances_

feature_names = X.columns

feature_importance = pd.Series(
    importance,
    index=feature_names
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)

plt.title("Feature Importance")
plt.xlabel("Importance Score")

plt.show()

# %%
sample = X_test[0].reshape(1, -1)

prediction = rf_model.predict(sample)

if prediction[0] == 1:
    print("\nPredicted Wine Quality: GOOD")
else:
    print("\nPredicted Wine Quality: BAD")


