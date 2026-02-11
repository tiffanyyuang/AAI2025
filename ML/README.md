# AAI2025 – Machine Learning Mini Project

This repository contains three machine learning examples demonstrating:

1. Regression  
2. Classification  
3. Clustering  

Each example uses Python and scikit-learn.

---

## 📁 Files

### 1) house_price_prediction.py
Linear Regression model that predicts house prices based on:
- Square footage  
- Location (one-hot encoded)

Outputs:
- Predicted price for a 2000 sqft house in Downtown  
- Model R² score  

---

### 2) customer_churn_prediction.py
Logistic Regression model that predicts probability of customer churn using:
- Age  
- Monthly usage  
- Purchase amount  
- Customer service calls  
- Region  

Outputs:
- Churn probability  
- At-risk classification (threshold = 0.5)  
- Accuracy, confusion matrix, and classification report  
- Feature coefficients  

---

### 3) customer_segmentation.py
K-Means clustering for customer segmentation using:
- Annual spending  
- Purchase frequency  
- Age  

Includes:
- Elbow method to choose K  
- Cluster assignments  
- Cluster means  
- Saves elbow_plot.png and customer_segments.csv  

---

## 📊 Libraries Used
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

---

## 📌 Assumptions
- Relationships between variables are approximately linear  
- Sample datasets are synthetic but realistic  
- Larger real-world datasets would improve performance  

---

## 🚀 Possible Improvements
- Use real datasets  
- Try Random Forest or Gradient Boosting  
- Add cross-validation  
- Hyperparameter tuning  

---

## ▶ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python ML/house_price_prediction.py
python ML/customer_churn_prediction.py
python ML/customer_segmentation.py
