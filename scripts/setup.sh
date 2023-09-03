#!/bin/bash

module load gcc arrow python httpproxy
virtualenv --no-download .env
source .env/bin/activate
pip install --no-index -r requirements/install.txt
mkdir packages
pip download --no-deps -r requirements/downloads.txt -d packages
pip install --no-deps packages/*
pip install -e .

