# 🏠 House Price Prediction using Linear Regression

## 📌 Overview
This project predicts house prices based on features such as area, bedrooms, bathrooms, and other property attributes using **Linear Regression**.

It demonstrates a complete Machine Learning workflow including data preprocessing, model training, evaluation, and visualization.

---

## 📂 Dataset Features

- area
- bedrooms
- bathrooms
- stories
- parking
- mainroad
- guestroom
- basement
- hotwaterheating
- airconditioning
- prefarea
- furnishingstatus
- price (Target Variable)

---

## ⚙️ Workflow

Data Loading → Data Cleaning → EDA → Feature Engineering → Model Training → Prediction → Evaluation → Visualization

---

## 🧠 Model Used

- Algorithm: Linear Regression  
- Library: Scikit-learn  
- Type: Supervised Regression Problem  

---

## 📊 Model Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📈 Visualizations

- Correlation Heatmap  
- Actual vs Predicted Plot  
- Residual Plot  
- Price vs Area Curve  

---

## 🔮 Sample Prediction

Input:
- Area = 5000
- Bedrooms = 3
- Bathrooms = 2

Output:
- Predicted Price ≈ 6,496,480

---

## 🚀 How to Run the Project
- python main.py
- pip install -r requirements.txt

## 📌 Project Structure

```
house-price-prediction/
│
├── Predicting_House_Prices.py
├── Housing.csv
├── README.md
└── requirements.txt
```

---

## 📊 Key Learnings

- Data preprocessing techniques  
- Handling categorical variables using encoding  
- Building Linear Regression model  
- Model evaluation techniques  
- Data visualization for insights  

---

## 📈 Conclusion

This project successfully demonstrates the use of **Linear Regression for predicting house prices** based on multiple input features. The model learns patterns from historical housing data and provides reasonably accurate price predictions.

From this project, we conclude that:

- House area is the most influential factor in determining price  
- Proper data preprocessing improves model performance significantly  
- Linear Regression works well as a baseline model for regression tasks  
- Feature engineering and encoding play a key role in model accuracy  
- Visualization helps in better understanding of data and model behavior  

Although the model performs well, its accuracy can be further improved using advanced algorithms such as **Ridge Regression, Lasso Regression, or Random Forest Regressor**.

---