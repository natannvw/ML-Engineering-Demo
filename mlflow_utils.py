import subprocess
from typing import Optional, Tuple, Union  # noqa: F401

import mlflow
from mlflow.entities.run_info import RunInfo  # noqa: F401
from mlflow.tracking import MlflowClient


def set_MlflowClient(
    mlflow_tracking_uri=None,
) -> MlflowClient:
    mlflow_client = MlflowClient(mlflow_tracking_uri)
    return mlflow_client


def set_experiment_client(experiment_name: str, mlflow_client: MlflowClient):
    experiment = mlflow_client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow_client.create_experiment(experiment_name)

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    return experiment_id


def set_mlflow(
    experiment_name="Test 1", mlflow_tracking_uri=None
) -> Tuple[int, MlflowClient]:
    mlflow_client = set_MlflowClient(mlflow_tracking_uri)
    experiment_id = set_experiment_client(experiment_name, mlflow_client)

    return experiment_id, mlflow_client


def start_mlflow_server(host="127.0.0.1", port=8080):
    """
    Start the MLflow tracking server on the local host with specified port and return the tracking URI.

    Parameters:
    - host (str): The host address to bind the server. Use '127.0.0.1' for localhost.
    - port (int): The port on which the MLflow server will listen.

    Returns:
    - str: The MLflow tracking URI.
    """
    command = ["mlflow", "server", "--host", host, "--port", str(port)]
    # Start the server in a subprocess
    subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    mlflow_tracking_uri = f"http://{host}:{port}"

    return mlflow_tracking_uri
