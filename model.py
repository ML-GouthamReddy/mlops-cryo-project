import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set MLflow tracking URI to a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Cryotherapy_Classification")

with mlflow.start_run():
    # Load data from the local data directory
    Cryo_Data = pd.read_csv("data/Cryotherapy.csv")

    X = Cryo_Data.iloc[:, :-1].values
    y = Cryo_Data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    # Train Model (Logistic Regression Only)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate metrics and save them to variables
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Log parameters
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("random_state", 27)

    # Save Model to the root directory
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
        
    # Print results to stdout for GitHub Actions logs
    print("Model trained successfully!")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Model saved as model.pkl and logged to MLflow")