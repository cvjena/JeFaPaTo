#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jefapato
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements.txt

pre-commit install
pre-commit autoupdate
