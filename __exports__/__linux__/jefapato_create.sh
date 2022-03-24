#!/bin/bash
eval "$(conda shell.bash hook)"
conda create -n jefapato -y
conda activate jefapato
conda install python=3.9 -y
conda install pip -y
python --version
python -m pip install -r requirements.txt
