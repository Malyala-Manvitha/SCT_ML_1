# House Price Prediction Using Linear Regression

## 📖 Project Overview
This project implements a **Linear Regression model** to predict house prices based on:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms

The project is developed as part of a **Skill Intelligence** task and demonstrates the application of supervised machine learning using Python.

---

## 🎯 Objective
To build a machine learning model that can accurately predict house prices using basic housing features and evaluate its performance using standard metrics.

---

## 🛠️ Technologies Used
- **Programming Language:** Python
- **IDE:** Visual Studio Code
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

---

## 📁 Project Structure
House_Price_Prediction/
│
├── dataset/
│ └── house_prices.csv
│
├── main.py
├── requirements.txt
├── README.md


---

## 📊 Dataset Description
The dataset is a CSV file containing the following columns:

| Column Name | Description |
|------------|-------------|
| sqft | Square footage of the house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| price | House price |

**Sample Data:**
sqft,bedrooms,bathrooms,price
1200,2,1,150000
1500,3,2,200000
1800,3,2,240000

---

## ⚙️ Installation & Setup

### Step 1: Clone or Download the Project
```bash
git clone <your-github-repo-link>
Or download the ZIP and extract it.
Step 2: Open Project in VS Code

Open Visual Studio Code

Click File → Open Folder

Select House_Price_Prediction

Step 3: Install Required Libraries

Run the following command in the VS Code terminal:
python -m pip install pandas numpy scikit-learn matplotlib
▶️ How to Run the Project

1.Open the terminal in VS Code

2.Make sure you are inside the project folder

3.Run:
python main.py
📈 Output

1.Model evaluation metrics:

2.Mean Squared Error (MSE)

3.R² Score

4.Model coefficients

5.Predicted house price based on user input

#Sample Output:

House Price Prediction System
==============================
Model Evaluation:
Mean Squared Error: 250000000.00
R² Score: 0.95

Enter square footage: 2000
Enter number of bedrooms: 3
Enter number of bathrooms: 2
Predicted House Price: ₹285000.00
🧠 Machine Learning Model

1.Algorithm: Linear Regression

2.Type: Supervised Learning

3.Evaluation Metrics:

4.Mean Squared Error

5.R² Score

✅ Conclusion

The Linear Regression model successfully predicts house prices using basic features. The project demonstrates fundamental machine learning concepts such as data preprocessing, model training, evaluation, and prediction.

📌 Author

Malyala Manvitha
Skill Intelligence Project

📜 License

This project is for educational purposes only.

---

