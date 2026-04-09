# 🚗 Used Car Price Predictor: A Machine Learning Regression Project

## 📌 Business Problem
Platforms like Cars24 or CarDekho need to evaluate the exact resale value of thousands of used cars daily. Relying on human estimates can lead to profit loss or customer dissatisfaction. The objective of this project is to build an automated, high-accuracy Machine Learning pipeline that predicts the exact selling price of a used car based on its features (Age, Kms Driven, Fuel Type, etc.).

## 🧠 Data Engineering & Preprocessing
Before feeding data to the model, several strict preprocessing steps were executed:
* **Feature Engineering:**  Converted the raw manufacturing `Year` into `Car_Age` (current year - manufacturing year) to provide a continuous numerical logic to the model. Dropped the redundant 'Car_Name' feature to avoid the curse of dimensionality.
* **Handling Categorical Data:** Applied One-Hot Encoding (`pd.get_dummies`) on text columns (Fuel_Type, Seller_Type, Transmission) while strictly enforcing `drop_first=True` to prevent the Multi-Collinearity / Dummy Variable Trap.

## 🏗️ Architecture: The "Base" vs "Advanced" Strategy
Instead of directly applying a heavy algorithm, I established a baseline to measure true business impact:

**1. The Baseline Model (Linear Regression)**
* Created a standard `Pipeline` with `StandardScaler` and `LinearRegression`.
* **Result:** Achieved a Base R2 Score of **84.89%**. Good, but not optimal.

**2. The Optimized VIP Pipeline (Random Forest + GridSearchCV)**
* Built a new pipeline utilizing the powerful `RandomForestRegressor`.
* Implemented `GridSearchCV` with **K-Fold Cross-Validation (cv=5)** to rigorously test the model and prevent overfitting.
* Set `n_jobs=-1` to utilize all CPU cores for faster hyperparameter tuning.
* **Hyperparameters Tuned:** Extracted the absolute best tree configuration (`max_depth: 30`, `n_estimators: 200`).

## 📈 Final Business Impact (Results)
By upgrading from a manual Linear Regression approach to a fully automated and tuned Random Forest Pipeline, the prediction accuracy surged significantly:
* **Final R2 Score:** **96.04%** 🚀
* **Conclusion:** The model can now explain 96% of the variance in used car prices, making it a highly reliable tool for real-world automated price estimation.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn (`Pipeline`, `GridSearchCV`, `RandomForestRegressor`, `LinearRegression`, `StandardScaler`), Metrics (`r2_score`).