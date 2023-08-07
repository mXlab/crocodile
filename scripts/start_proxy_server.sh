#!/bin/bash

module load gcc arrow postgresql python
source .env/bin/activate

flask --app scripts/proxy_server.py run --host=0.0.0.0 --port=8080

