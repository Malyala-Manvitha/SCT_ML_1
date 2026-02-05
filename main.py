# House Price Prediction using Linear Regression
# Skill Intelligence Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    """Load the house price dataset"""
    try:
        data = pd.read_csv("dataset/house_prices.csv")
        return data
    except FileNotFoundError:
        print("Dataset file not found. Please check the path.")
        exit()


def preprocess_data(data):
    """Split features and target"""
    X = data[['sqft', 'bedrooms', 'bathrooms']]
    y = data['price']
    return X, y


def train_model(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation:")
    print("------------------")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return predictions


def display_coefficients(model, feature_names):
    """Display model coefficients"""
    print("\nModel Coefficients:")
    print("-------------------")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")


def predict_new_house(model):
    """Predict price for a new house"""
    sqft = float(input("\nEnter square footage: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))

    new_data = np.array([[sqft, bedrooms, bathrooms]])
    predicted_price = model.predict(new_data)

    print(f"\nPredicted House Price: ₹{predicted_price[0]:.2f}")


def main():
    print("House Price Prediction System")
    print("==============================")

    data = load_data()
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    display_coefficients(model, X.columns)

    predict_new_house(model)


if __name__ == "__main__":
    main()
