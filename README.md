# ✈️ Flight Price Prediction

This project predicts the prices of airline tickets based on several flight-related factors such as airline, source, destination, duration, and total stops.  
It uses machine learning models to learn from historical flight price data and estimate prices for future trips.

## 🧠 Overview

Flight ticket prices are dynamic and influenced by several variables like travel duration, stops, and airlines.  
This project applies **data preprocessing**, **exploratory data analysis (EDA)**, and **machine learning regression models** to predict flight prices.

---

## 📂 Dataset

The dataset (usually named `Data_Train.xlsx`) contains flight information with columns such as:

| Column | Description |
|---------|-------------|
| Airline | Name of the airline |
| Date_of_Journey | Date of flight |
| Source | Departure city |
| Destination | Arrival city |
| Route | Route taken by the flight |
| Dep_Time | Departure time |
| Arrival_Time | Arrival time |
| Duration | Total travel duration |
| Total_Stops | Number of stops |
| Additional_Info | Other details |
| Price | Target variable (Ticket price) |

> Note: You can download the dataset from Kaggle or use the one provided in this repository (if available).

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Converting date/time columns to numerical format
   - Extracting features from `Date_of_Journey`
   - Encoding categorical variables using `LabelEncoder` and `OneHotEncoder`

2. **Exploratory Data Analysis (EDA)**
   - Visualizing distributions of flight prices
   - Checking relationships between airline, duration, and stops

3. **Feature Engineering**
   - Transforming `Duration` into total minutes/hours
   - Removing redundant or less useful columns
   - Creating dummy variables for categorical columns

4. **Model Building**
   - Splitting data into train/test sets
   - Training models such as:
     - `RandomForestRegressor`
     - `DecisionTreeRegressor`
     - `XGBoostRegressor`
   - Evaluating models using metrics like R² and RMSE

5. **Model Saving**
   - Exporting the trained model using `pickle` or `joblib`

---

## 🤖 Modeling

Example of model training code:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Save the model
pickle.dump(model, open('flight_rf.pkl', 'wb'))
