#!/bin/bash
cd -- "$(dirname "$BASH_SOURCE")"

eval "$(conda shell.bash hook)"
conda create -n jefapato-prod -y
conda activate jefapato-prod
conda install python=3.11 -y
conda install pip -y
conda install -c conda-forge opencv=4.7 -y
python --version
python -m pip install -r requirements.txt
