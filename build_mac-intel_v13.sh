#!/bin/bash

# NOTE if you get some libffi.8.dylib error, a likely reason is that the virtual environment
# is based on the python version of conda (e.g. miniconda) and not the python version of brew
# therfore deactivate conda COMPLETELY, delete the virtual environment, and rerun this script

# this is needed that python gets the correct platform version on mac big sur
export SYSTEM_VERSION_COMPAT=0

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
# if brew ls --versions create-dmg > /dev/null; then
#     echo "create-dmg is installed"
# else
#     echo "create-dmg is not installed"
#     echo "installing create-dmg"
#     brew install create-dmg
# fi

# check if python3 is installed, else prompt the user to install the UNIVERSAL installer
# for python 3 we need at least version 3.10.11
if python3 -V > /dev/null; then
    echo "python3 is installed"
else
    echo "python3 is not installed"
    echo "installing python3"
    echo "install the UNIVERSAL installer from https://www.python.org/downloads/"
    exit 1
fi

# check if the virtual environment exists
# if not exists, create a virtual environment
if [ -d "venv-mac-intel" ]; then
    echo "virtual environment exists"
    source venv-mac-intel/bin/activate
else
    echo "virtual environment does not exist"
    echo "creating virtual environment"
    # create a virtual environment
    python3 -m venv venv-mac-intel 
    source venv-mac-intel/bin/activate
fi

python3 -c "import platform; print(platform.platform())"
# update pip else some installation scripts might fail!
arch -x86_64 python3 -m pip install --upgrade pip setuptools wheel
arch -x86_64 python3 -m pip install --upgrade --force-reinstall -r requirements-dev.txt 
arch -x86_64 python3 -m pip install --upgrade --force-reinstall -r requirements.txt

rm -rf build
arch -x86_64 python setup.py py2app --arch=x86_64
mkdir -p dist/intel
mv dist/JeFaPaTo.app dist/intel/JeFaPaTo.app

mkdir -p dist/dmg
# create a dmg file, requires create-dmg from brew to be installed
create-dmg \
    --volname JeFaPaTo_intel \
    --volicon frontend/assets/icons/icon.icns \
    --window-pos 200 120 \
    --window-size 800 400 \
    --icon-size 100 \
    --icon "JeFaPaTo.app" 200 190 \
    --hide-extension "JeFaPaTo.app" \
    --app-drop-link 600 185 \
    --no-internet-enable \
    "JeFaPaTo_intel.dmg" \
    dist/intel

mv JeFaPaTo_intel.dmg dist/dmg/JeFaPaTo_intel.dmg