# House Price Prediction - Linear Regression
# Data Source:
# https://www.kaggle.com/datasets/shivachandel/kc-house-data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = {
    "square_footage": [1500,2000,1800,2500,2200,1700,3000,1900,2100,2600,
                       1600,2300,2800,1400,2700,2400,3100,1750,1950,2250],
    "location": ["Downtown","Suburb","Downtown","Rural","Suburb",
                 "Downtown","Rural","Suburb","Downtown","Rural",
                 "Suburb","Downtown","Rural","Suburb","Downtown",
                 "Suburb","Rural","Downtown","Suburb","Downtown"],
    "price": [300000,350000,320000,280000,360000,
              310000,400000,340000,330000,290000,
              315000,370000,420000,295000,390000,
              365000,450000,325000,345000,375000]
}

df = pd.DataFrame(data)

X = df[["square_footage","location"]]
y = df["price"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), ["location"])],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

new_house = pd.DataFrame({
    "square_footage":[2000],
    "location":["Downtown"]
})

prediction = model.predict(new_house)

print("Predicted price:", round(prediction[0],2))
print("Model R2 Score:", model.score(X_test, y_test))

