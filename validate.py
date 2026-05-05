import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

print("Iniciando validación...")

mlflow.set_tracking_uri("file:./mlruns")

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("mlflow-deploy-experiment")

if experiment is None:
    print("No se encontró el experimento")
    exit()

runs = client.search_runs(experiment.experiment_id)

if len(runs) == 0:
    print("No hay runs registrados")
    exit()

latest_run = runs[0]
run_id = latest_run.info.run_id

print(f"Run encontrado: {run_id}")

model_uri = f"runs:/{run_id}/modelo"

print("Cargando modelo...")
model = mlflow.sklearn.load_model(model_uri)

print("Modelo cargado correctamente")

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

X = data.drop("species", axis=1)
y = data["species"]

preds = model.predict(X)

acc = accuracy_score(y, preds)

print(f"Validation Accuracy: {acc}")