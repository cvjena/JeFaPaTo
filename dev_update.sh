#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jefapato
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements.txt

pre-commit install
pre-commit autoupdate

curl -O https://google.github.io/styleguide/pylintrc
sed -i "s/indent-string='  '/indent-string='    '/" pylintrc
sed -i "s/max-line-length=80/max-line-length=88/" pylintrc
