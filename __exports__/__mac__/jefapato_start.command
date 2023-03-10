#!/bin/bash
cd -- "$(dirname "$BASH_SOURCE")"

eval "$(conda shell.bash hook)"
conda activate jefapato
python main.py
