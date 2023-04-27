#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jefapato-prod
python -m pip install -r requirements.txt
