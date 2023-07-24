#!/bin/bash

module load gcc arrow postgresql python
source .env/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="crocodile-credentials.json"

port=5000

while getopts p: flag
do
    case "${flag}" in
        p) port=${OPTARG};;
    esac
done

mlflow server \
  --backend-store-uri postgresql://postgres:crocodile@34.28.119.2/postgres \
  --artifacts-destination gs://crocodile-333216.appspot.com/models \
  --host 0.0.0.0 \
  --port $port