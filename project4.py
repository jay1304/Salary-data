import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("salary_data.csv")

# Input Feature
X = df[['Experience']]

# Target Variable
y = df['Salary']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Error Calculation
mae = mean_absolute_error(y_test, predictions)

print("Actual Salary:")
print(y_test.values)

print("\nPredicted Salary:")
print(predictions)

print("\nMean Absolute Error:")
print(round(mae, 2))

# Predict Salary for New Employee
experience = 5

new_data = pd.DataFrame({
    'Experience': [experience]
})

new_prediction = model.predict(new_data)

print(
    f"\nPredicted Salary for {experience} years of experience:"
)
print(round(new_prediction[0], 2))