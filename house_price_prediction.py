
import pandas as pd
import numpy as np

# Create dataset
data = {
    "House_ID": range(1, 21),
    "Area": np.random.choice(["Urban", "Suburban", "Rural"], size=20),
    "Size": np.random.randint(800, 3000, size=20),
    "Bedrooms": np.random.randint(1, 6, size=20),
    "Price_per_sqft": np.round(np.random.uniform(50, 200, size=20), 2),
    "Price": None
}

# Calculate Price
df = pd.DataFrame(data)
df["Price"] = (df["Size"] * df["Price_per_sqft"]).round(2)

# Save as CSV
df.to_csv("HousePrices.csv", index=False)
df
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("HousePrices.csv")

# Overview
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize data
sns.pairplot(df, hue="Area")
plt.show()

# Distribution of Price
sns.histplot(df["Price"], kde=True, bins=10)
plt.title("Price Distribution")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode Area
le = LabelEncoder()
df["Area"] = le.fit_transform(df["Area"])

# Normalize numerical features
scaler = StandardScaler()
df[["Size", "Price_per_sqft", "Price"]] = scaler.fit_transform(df[["Size", "Price_per_sqft", "Price"]])

# Split data
X = df[["Area", "Size", "Bedrooms", "Price_per_sqft"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Area vs Price
sns.boxplot(x="Area", y="Price", data=df)
plt.title("Area vs Price")
plt.show()

# Bedrooms vs Price
sns.scatterplot(x="Bedrooms", y="Price", data=df)
plt.title("Bedrooms vs Price")
plt.show()
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}

# Train and evaluate each model
results = []
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save results
    results.append({"Model": name, "MSE": mse, "R-squared": r2})

# Create a DataFrame to display results
results_df = pd.DataFrame(results)
print(results_df)
import joblib

# Save model
joblib.dump(model, "house_price_model.pkl")
