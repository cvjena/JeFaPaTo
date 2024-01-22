#!/bin/bash

# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"
conda activate jefapato
rm -r build
rm -r dist
pyinstaller --console --onefile --name JeFaPaTo --add-data frontend:frontend --add-data examples:examples --icon "frontend\assets\icons\icon.ico" main.py
mv dist/JeFaPaTo dist/JeFaPaTo_linux