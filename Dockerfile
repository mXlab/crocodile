
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./requirements/server.txt ./requirements/server.txt

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements/server.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD mlflow server --backend-store-uri postgresql+pg8000://postgres:crocodile@/postgres?unix_sock=/cloudsql/crocodile-333216:us-central1:crocodile-mlflow/.s.PGSQL.5432 --artifacts-destination gs://crocodile-333216.appspot.com/models --port 8080 --host 0.0.0.0