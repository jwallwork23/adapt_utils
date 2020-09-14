#!/bin/bash

# ====================================================================== #
# Bash script for installing PyADOL-C, Python wrapper for the C++        #
# automatic differentation tool, ADOL-C.                                 #
#                                                                        #
# Basic dependencies:                                                    #
#  * libtool                                                             #
#  * libboost-all-dev                                                    #
#  * wget                                                                #
#                                                                        #
# Pip dependencies for testing:                                          #
#  * matplotlib                                                          #
#  * nose                                                                #
#                                                                        #
# The virtual environment should be activated.                           #
#                                                                        #
#                                      Joe Wallwork, 14th September 2020 #
# ====================================================================== #

# Check existence of SOFTWARE environment variable
if [ ! -e "$SOFTWARE" ]; then
    echo "SOFTWARE environment variable $SOFTWARE does not exist."
    exit 1
fi

# Check virtual environment is active
if [ ! -e "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is not active."
    exit 1
fi

# Install dependencies
sudo apt-get update -y
sudo apt-get install -y libtool libboost-all-dev wget

# Clone repo from GitHub
cd $SOFTWARE
mkdir PyADOL-C
cd PyADOL-C
git clone https://github.com/b45ch1/pyadolc.git
export PYTHONPATH=$SOFTWARE/PyADOL-C:$PYTHONPATH
cd pyadolc

# Install ADOL-C
./bootstrap.sh

# Apply compile flags patch for Python3
git apply $SOFTWARE/pyadolc_python3.patch

# Build, install
CC=gcc CXX=g++ python3 setup.py build
python3 setup.py install

# KNOWN INSTALL BUG:
#   If you are using Python3.6 then you may need to remove the NULL return statement in
#   `$VIRTUAL_ENV/lib/python3.6/site-packages/numpy/core/include/numpy/__multiarray_api.h`.

# Test (outside of install directory)
cd $SOTWARE
python3 -m pip install matplotlib nose
python3 -c "import adolc; adolc.test()"
