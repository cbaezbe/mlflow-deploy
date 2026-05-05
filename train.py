import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# Forzar ruta local
mlflow.set_tracking_uri("file:./mlruns")

# Crear experimento (si no existe)
mlflow.set_experiment("mlflow-deploy-experiment")

# Dataset externo
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

X = data.drop("species", axis=1)
y = data["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
model = LogisticRegression(max_iter=200)

with mlflow.start_run():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logs
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    # Signature
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.head(2)

    # Guardar modelo
    mlflow.sklearn.log_model(
        model,
        name="modelo",
        signature=signature,
        input_example=input_example
    )

    print(f"Modelo entrenado con accuracy: {acc}")