#!/bin/bash

# TODO
# - make more general
# - add version numbers
# - switch to universal2 (check which modules are the issues)

# check if brew is installed
# if not abort
if brew -v > /dev/null; then
    echo "brew is installed"
else
    echo "brew is not installed"
    echo "aborting, install first @ https://brew.sh/"
    exit 1
fi

# check if create-dmg is installed
# if not install it
if brew ls --versions create-dmg > /dev/null; then
    echo "create-dmg is installed"
else
    echo "create-dmg is not installed"
    echo "installing create-dmg"
    brew install create-dmg
fi


# check if python3.10 is installed (via brew)
if brew ls --versions python@3.10 > /dev/null; then
    echo "python3.10 is installed"
else
    echo "python3.10 is not installed"
    echo "installing python3.10"
    brew install python@3.10
fi

# check mac architecture
if [[ $(uname -m) == "arm64" ]]; then
    echo "mac architecture is arm64"
    # check if the virtual environment exists
    # if not exists, create a virtual environment
    if [ -d "venv-m1-arm64" ]; then
        echo "virtual environment exists"
        source venv-m1-arm64/bin/activate
    else
        echo "virtual environment does not exist"
        echo "creating virtual environment"
        # create a virtual environment
        python3.10 -m venv venv-m1-arm64
        source venv-m1-arm64/bin/activate

        # install dependencies
        python -m pip install -r requirements-dev.txt
        python -m pip install -r requirements.txt
    fi
    rm -rf build
    rm -rf dist
    python setup.py py2app --arch=universal2
    # create a dmg file, requires create-dmg from brew to be installed
    rm JeFaPaTo_M1-arm64.dmg 
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
        " JeFaPaTo_M1-arm64.dmg" \
        dist

else
    echo "mac architecture is not arm64"
    # check if the virtual environment exists
    if [ -d "venv-intel-x86_64" ]; then
        echo "virtual environment exists"
        source venv-intel-x86_64/bin/activate
    else
        echo "virtual environment does not exist"
        echo "creating virtual environment"
        # create a virtual environment
        python3.10 -m venv venv-intel-x86_64
        source venv-intel-x86_64/bin/activate

        # install dependencies
        python -m pip install -r requirements-dev.txt
        python -m pip install -r requirements.txt
    fi

    rm -rf build
    rm -rf dist
    python setup.py py2app --arch=x86_64

    # create a dmg file, requires create-dmg from brew to be installed
    rm JeFaPaTo_Intel-x86_64.dmg
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
        " JeFaPaTo_Intel-x86_64.dmg" \
        dist
fi
