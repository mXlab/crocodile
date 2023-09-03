import mlflow

experiment_name = "test"
remote_server_uri = "https://crocodile-gqhfy6c73a-uc.a.run.app/"

mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    pass
