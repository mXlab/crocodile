
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim


# Install Postgres client and dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./requirements/server.txt ./requirements/server.txt
COPY ./crocodile-credentials.json ./crocodile-credentials.json

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements/server.txt

ENV GOOGLE_APPLICATION_CREDENTIALS="./crocodile-credentials.json"

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD GOOGLE_APPLICATION_CREDENTIALS="./crocodile-credentials.json" mlflow server --backend-store-uri postgresql+psycopg2://postgres:crocodile@/postgres?host=/cloudsql/crocodile-333216:us-central1:crocodile-mlflow --artifacts-destination gs://crocodile-333216.appspot.com/models --host 0.0.0.0 --port 8080