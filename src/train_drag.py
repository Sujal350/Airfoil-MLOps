import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


# Load Dataset
df = pd.read_excel("data/Airfoil data.xlsx")

# Features and Target
X = df[['Max Camber', 'Max Camber Position', 'Thickness', 'AOA']]
y = df['CD']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = mean_squared_error(
    y_train,
    y_train_pred
) ** 0.5

test_rmse = mean_squared_error(
    y_test,
    y_test_pred
) ** 0.5

print("\n===== Drag Model Results =====")
print(f"Train R2  : {train_r2:.4f}")
print(f"Test R2   : {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test RMSE : {test_rmse:.6f}")

# Save Model
joblib.dump(
    model,
    "models/gradient_boosting_cd_model.pkl"
)

print("\nModel saved successfully.")