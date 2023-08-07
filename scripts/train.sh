#!/bin/bash

module load gcc arrow python
source .env/bin/activate
python scripts/train.py $@