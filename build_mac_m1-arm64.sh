#!/bin/bash

# check if the virtual environment exists
# if not exists, create a virtual environment

if [ -d "venv-jefapato-mac-m1-arm64" ]; then
    echo "virtual environment exists"
    source venv-jefapato-mac-m1-arm64/bin/activate

else
    echo "virtual environment does not exist"
    echo "creating virtual environment"
    # create a virtual environment
    python3.10 -m venv venv-jefapato-mac-m1-arm64
    source venv-jefapato-mac-m1-arm64/bin/activate

    # install dependencies
    python -m pip install -r requirements-dev.txt
    python -m pip install -r requirements.txt
fi

rm -rf build dist
python setup.py py2app

# create a dmg file, requires create-dmg from brew to be installed

rm JeFaPaTo.dmg
create-dmg \
    --volname JeFaPaTo \
    --volicon frontend/assets/icons/icon.icns \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 100 \
    --icon "JeFaPaTo.app" 200 190 \
    --hide-extension "JeFaPaTo.app" \
    --app-drop-link 600 185 \
    --no-internet-enable \
    "JeFaPaTo.dmg" \
    dist