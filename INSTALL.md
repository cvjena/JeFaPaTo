# Installation

This section describes how to install JeFaPaTo from source. If you want to use the precompiled binaries, please refer to the [README.md](../README.md).

The general installation process is the same for all platforms as we highly recommend the usage of `conda` to manage the dependencies. Even though `conda` is not required and `venv` would be sufficient enough, it is highly recommended to use it.
If you decide to use `venv`, please refer to the [venv documentation](https://docs.python.org/3/library/venv.html) for further instructions and manually install the dependencies listed in the [requirements.txt](requirements.txt) file and the [requirements-dev.txt](requirements-dev.txt).
If you have issues with the `venv` installation, you can find some inspiration in the [build_mac-intel_v13.sh](build_mac-intel_v13.sh) script as we used it to build the binaries for macOS to avoid conflicts with the `conda` python.

## Prerequisites

JeFaPaTo utilizes only packages available through the `python package index (PyPI)`. We did our best to enforce correct versioning of the dependencies, but we cannot guarantee that the installation will work with newer versions of the packages. Also for sending notifications via `plyer` some systems need further dependencies but require we hope to got them all. Please refer to the [plyer documentation](https://plyer.readthedocs.io/en/latest/) for further instructions if you encounter any issues, and please let us know via the [issue tracker](https://github.com/cvjena/JeFaPaTo/issues/new).

If your dbus is not configured correctly, you might have to install the following packages:

```bash
sudo apt install build-essential libdbus-glib-1-dev libgirepository1.0-dev
```

We currently use `Python 3.10` to develop JeFaPaTo, and older versions are not recommended as make high usage of the new typing features. We recommend using `Python 3.10` or newer, but we cannot guarantee that it will work with older versions.

## Local installation

The installation, without using our precompiled binaries, is the same for all platforms.
We assume that you have `conda` installed and configured. If not, please refer to the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for further instructions.

You can either run the `dev_init.sh` script or follow the instructions below.

```bash
bash dev_init.sh
```

or manually:

```bash
# Clone the repository
git clone https://github.com/cvjena/JeFaPaTo.git
cd JeFaPaTo
conda create -n jefapato python=3.10 pip -y
conda activate jefapato
# not 100% necessary but recommended if you want to build the binaries
conda install libpython-static -y

# Install the dependencies, including the development dependencies
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Usage

After the installation, you can run JeFaPaTo with the following command:

```bash
# assuming you activated the conda environment :^)
# else conda activate jefapato
python main.py
```

The GUI should open, intermediate information should be visible in the terminal, and you can start using JeFaPaTo.

## Build the binaries

This section describes how to build the binaries for the different platforms. We recommend using the precompiled binaries, but if you want to build them yourself, you can follow the instructions below, but please note that we do not guarantee that the binaries will work on your system.

The binaries are built with `pyinstaller`, and each platform has its own script to build the binaries. The scripts are fully automated but not yet hooked into the CI/CD pipeline. Hence, we currently build the binaries manually and upload them to the [release page]().
If you want to build the binaries yourself, you can use the following scripts:

### Windows 11 (x64)

This script has to be executed in a `Windows 11` machine.

```bash
.\build_windows-11.bat
```

### macOS v13+ (x64)

This script has to be executed in a `macOS v13+` machine, and the `universal version` is only supported on `Apple Silicon`.
The `Intel` version is only supported on `Intel` machines and `Apple Silicon` machines with `Rosetta 2` installed.

```bash
# for Apple Silicon and Intel
./build_mac-universal2_v13.sh
# for Intel only
./build_mac-intel_v13.sh
```

### macOS v10

This version is only supported on the branch `intel_macosx10_mediapipe_v0.9.0` and is only kept alive to support older macOS versions. We highly recommend using the latest version of macOS.

```bash
./build_mac-intel_v10.sh
```

### Linux (x64)

This script has to be executed in a `Linux` machine, in our case `Ubuntu 22.04`.

```bash
./build_linux.sh
```
