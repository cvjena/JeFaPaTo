#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jefapato-intel-mediapipe-v9
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements.txt

# pre-commit install
# pre-commit autoupdate
