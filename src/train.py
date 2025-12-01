import pandas as pd
import pickle
import json
import mlflow
import mlflow.sklearn
import mlflow.models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import yaml
import os

# Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

model_type = params["train"]["model_type"]
random_state = params["train"]["random_state"]
n_estimators = params["train"]["n_estimators"]

# Загрузка данных
train = pd.read_csv("data/prepared/train.csv")
test = pd.read_csv("data/prepared/test.csv")

X_train = train.drop(columns=["variety"])
y_train = train["variety"]
X_test = test.drop(columns=["variety"])
y_test = test["variety"]

# Select model
if model_type == "logistic_regression":
    model = LogisticRegression(max_iter=200, random_state=random_state)
elif model_type == "random_forest":
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
else:
    raise ValueError(f"Unsupported model_type: {model_type}")

# MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Лог параметров и метрик
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("random_state", random_state)
    if model_type == "random_forest":
        mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)

    signature = mlflow.models.infer_signature(X_train, y_pred)
    input_example = X_train.iloc[:3]  # или .head(3)

    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
    )	

    metrics = { "accuracy": acc, }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Сохранение модели
    os.makedirs("data/models", exist_ok=True)
    with open("data/models/model.pkl", "wb") as f:
        pickle.dump(model, f)
        mlflow.log_artifact("data/models/model.pkl")
