#!/bin/bash

module load python
virtualenv --no-download .env
source .env/bin/activate
pip install --no-index --upgrade pip
mkdir packages
pip download --no-deps -r requirements/downloads.txt -d packages
pip install --no-deps packages/*
pip install --no-index -r requirements/beluga.txt
pip freeze --local > requirements/local.txt