# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")

# %%
apps = pd.read_csv("apps.csv")
reviews = pd.read_csv("user_reviews.csv")

print(apps.shape)
print(reviews.shape)

# %%
print(apps.head())
print(apps.info())

# %%
apps = apps.drop(columns=['Unnamed: 0'])

apps = apps[[
    'App',
    'Category',
    'Rating',
    'Reviews',
    'Size',
    'Installs',
    'Type',
    'Price',
    'Content Rating',
    'Genres',
    'Last Updated',
    'Current Ver',
    'Android Ver'
]]

apps.head()

# %%
apps.drop_duplicates(inplace=True)
reviews.drop_duplicates(inplace=True)

# %%
apps["Rating"] = pd.to_numeric(
    apps["Rating"],
    errors="coerce"
)

# %%
apps["Reviews"] = pd.to_numeric(
    apps["Reviews"],
    errors="coerce"
)

# %%
apps["Installs"] = (
    apps["Installs"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)
)

apps["Installs"] = pd.to_numeric(
    apps["Installs"],
    errors="coerce"
)

# %%
apps["Price"] = (
    apps["Price"]
    .astype(str)
    .str.replace("$", "", regex=False)
)

apps["Price"] = pd.to_numeric(
    apps["Price"],
    errors="coerce"
)

# %%
def convert_size(size):

    if pd.isna(size):
        return np.nan

    size = str(size)

    if size == "Varies with device":
        return np.nan

    if "M" in size:
        return float(size.replace("M", ""))

    if "k" in size:
        return float(size.replace("k", "")) / 1024

    return np.nan


apps["Size"] = apps["Size"].apply(convert_size)

# %%
apps["Last Updated"] = pd.to_datetime(
    apps["Last Updated"],
    errors="coerce"
)

# %%
numeric_cols = [
    "Rating",
    "Reviews",
    "Size",
    "Installs",
    "Price"
]

for col in numeric_cols:
    apps[col].fillna(
        apps[col].median(),
        inplace=True
    )

# %%
apps["Days_Since_Update"] = (
    pd.Timestamp.today() -
    apps["Last Updated"]
).dt.days

# %%
apps["Days_Since_Update"].fillna(
    apps["Days_Since_Update"].median(),
    inplace=True
)

# %%
apps["Popularity_Score"] = (
    np.log1p(apps["Reviews"]) +
    np.log1p(apps["Installs"])
)

# %%
apps["Engagement"] = (
    apps["Reviews"] /
    apps["Installs"]
)

# %%
apps["Revenue"] = (
    apps["Price"] *
    apps["Installs"]
)

# %%
plt.figure(figsize=(12,6))

apps["Category"].value_counts().head(15).plot(
    kind="bar"
)

plt.title("Top Categories")
plt.show()

# %%
plt.figure(figsize=(8,5))

sns.histplot(
    apps["Rating"],
    bins=20,
    kde=True
)

plt.title("Rating Distribution")
plt.show()

# %%
top_rated = (

    apps.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)

)

print(top_rated.head(10))

# %%
corr_cols = [

    "Rating",
    "Reviews",
    "Size",
    "Installs",
    "Price",
    "Popularity_Score"

]

plt.figure(figsize=(10,8))

sns.heatmap(
    apps[corr_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()

# %%
reviews["Sentiment"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Reviews Distribution")
plt.show()

# %%
sentiment_summary = reviews.groupby("App").agg({

    "Sentiment_Polarity":"mean",
    "Sentiment_Subjectivity":"mean"

}).reset_index()

final_df = pd.merge(

    apps,
    sentiment_summary,
    on="App",
    how="left"

)

# %%
num_cols = final_df.select_dtypes(include=np.number).columns
final_df[num_cols] = final_df[num_cols].fillna(0)

# %%
print(final_df.isnull().sum())

# %%
final_df.dropna(subset=["Current Ver", "Android Ver"], inplace=True)

# %%
print(final_df.isnull().sum())

# %%
features = [

    "Reviews",
    "Size",
    "Installs",
    "Price",
    "Popularity_Score",
    "Engagement",
    "Days_Since_Update"

]

X = final_df[features]

y = final_df["Rating"]

# %%
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# %%
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

# %%
print("MAE:",
      mean_absolute_error(y_test,pred))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test,pred)))

print("R2:",
      r2_score(y_test,pred))

# %%
final_df["Popular"] = np.where(

    final_df["Installs"] >= 1000000,
    1,
    0

)

# %%
X = final_df[features]

y = final_df["Popular"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(
    classification_report(
        y_test,
        pred
    )
)

# %%
cluster_data = final_df[

    [
        "Rating",
        "Reviews",
        "Installs",
        "Price"
    ]

]

# %%
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

final_df["Cluster"] = kmeans.fit_predict(
    cluster_data
)

# %%
pca = PCA(n_components=2)

components = pca.fit_transform(
    cluster_data
)

plt.figure(figsize=(10,6))

plt.scatter(

    components[:,0],
    components[:,1],
    c=final_df["Cluster"]

)

plt.title("App Clusters")
plt.show()

# %%
text = " ".join(

    reviews["Translated_Review"]
    .dropna()
    .astype(str)

)

wordcloud = WordCloud(

    width=1200,
    height=600,
    background_color="white"

).generate(text)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()


