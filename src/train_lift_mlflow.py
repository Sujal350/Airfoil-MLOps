import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


# Start MLflow Run
with mlflow.start_run():

    # Load Dataset
    df = pd.read_excel("data/Airfoil data.xlsx")

    # Features and Target
    X = df[['Max Camber', 'Max Camber Position', 'Thickness', 'AOA']]
    y = df['CL']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Parameters
    C = 10
    gamma = "scale"
    epsilon = 0.01

    # Log Parameters
    mlflow.log_param("C", C)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("epsilon", epsilon)

    # Model
    model = SVR(
        kernel="rbf",
        C=C,
        gamma=gamma,
        epsilon=epsilon
    )

    # Training
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Log Metrics
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)

    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)

    # Save Model
    joblib.dump(model, "models/svr_lift_model.pkl")
    joblib.dump(scaler, "models/lift_scaler.pkl")

    # Log Model
    mlflow.sklearn.log_model(
        model,
        artifact_path="lift_model"
    )

    print("\n===== Lift Model Results =====")
    print(f"Train R2 : {train_r2:.4f}")
    print(f"Test R2  : {test_r2:.4f}")