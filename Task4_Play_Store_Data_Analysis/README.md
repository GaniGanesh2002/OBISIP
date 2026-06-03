# 📱 Google Play Store Data Analysis & Machine Learning Project

## 📊 Project Overview

This project analyzes Google Play Store app data and user reviews to understand app market trends, user behavior, and factors affecting app success. It also builds machine learning models to predict app rating and popularity.

The project combines:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Sentiment Analysis (NLP)
- Feature Engineering
- Machine Learning Models
- Data Visualization

---

## 🎯 Objectives

- Understand Android app market trends
- Analyze user ratings and installs
- Study user sentiments from reviews
- Identify factors affecting app success
- Predict app rating using ML models
- Predict app popularity (1M+ installs)

---

## 📂 Datasets Used

### 1. apps.csv 📱
Contains app information such as:
- App name
- Category
- Rating
- Reviews
- Installs
- Size
- Price
- Type
- Content Rating
- Last Updated

### 2. user_reviews.csv 💬
Contains user feedback:
- App name
- Review text
- Sentiment (Positive/Negative/Neutral)
- Sentiment Polarity
- Sentiment Subjectivity

---

## 🧹 Data Preprocessing

- Removed duplicate values
- Converted data types (Installs, Price, Reviews)
- Handled missing values
- Converted app size into numeric format
- Cleaned review dataset
- Merged both datasets using App name

---

## 🛠 Feature Engineering

New features created:
- Popularity Score
- Engagement Rate
- Revenue Estimation
- Days Since Last Update

---

## 📊 Exploratory Data Analysis (EDA)

Key visualizations:
- App category distribution
- Rating distribution
- Installs vs Categories
- Correlation heatmap
- Sentiment distribution
- Word cloud of user reviews

---

## 💬 Sentiment Analysis

User reviews were analyzed to determine:
- Positive feedback
- Negative feedback
- Neutral feedback

Insights:
- Most users give positive feedback
- Ads and bugs are common complaints
- User experience strongly affects ratings

---

## 🤖 Machine Learning Models

### 1. Rating Prediction Model (Regression)
- Algorithm: Random Forest Regressor
- Goal: Predict app rating
- Features used:
  - Reviews
  - Installs
  - Size
  - Price
  - Popularity Score

---

### 2. Popularity Prediction Model (Classification)
- Algorithm: Random Forest Classifier
- Goal: Predict if app will reach 1M+ installs
- Output: Popular / Not Popular

---

## 📈 Clustering (Unsupervised Learning)

- Algorithm: K-Means Clustering
- Purpose: Group similar apps
- Features:
  - Rating
  - Installs
  - Reviews
  - Price

---

## 📉 Visualization Tools Used

- Matplotlib
- Seaborn
- WordCloud
- PCA (for clustering visualization)

---

## 🧠 Key Insights

- Free apps dominate the Play Store
- Games and Family categories are most popular
- Higher installs often lead to better ratings
- User sentiment strongly impacts app rating
- Regular updates improve app performance

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLP (Sentiment Analysis)
- Machine Learning

---

## 🚀 Future Improvements

- Deploy Streamlit dashboard
- Use Deep Learning (LSTM/BERT) for sentiment analysis
- Add real-time app prediction system
- Build API for rating prediction

---

## 📌 Conclusion

This project demonstrates how data analytics and machine learning can be used to understand app market behavior and predict app success using historical data.

---
