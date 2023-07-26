#!/bin/bash

# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"
conda create -n jefapato -y
conda activate jefapato
conda install python=3.10 pip -y
conda install libpython-static
python --version

# use pip as a module in the install python. this way we can avoid using the
# wrong pip installed on the system
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements.txt

# pre-commit install
# pre-commit autoupdate
