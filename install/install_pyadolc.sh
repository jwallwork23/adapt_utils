#!/bin/bash

# ====================================================================== #
# Bash script for installing the PyADOL-C AD tool.                       #
#                                                                        #
# Basic dependencies:                                                    #
#  * autotools                                                           #
#  * boost-python                                                        #
#  * libtool                                                             #
#  * libboost-all_dev                                                    #
#                                                                        #
# Dependencies for testing:                                              #
#  * matplotlib                                                          #
#  * nose                                                                #
#                                                                        #
# The virtual environment should be activated.                           #
#                                                                        #
#                                      Joe Wallwork, 14th September 2020 #
# ====================================================================== #

# Check virtual environment is active
if [ ! -e "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is not active."
    exit 1
fi

# Install dependencies
sudo apt-get install -y boost-python
sudo apt-get install -y autotools libtool libboost-all-dev

# Clone repo from GitHub
cd $SOFTWARE
mkdir PyADOL-C
cd PyADOL-C
git clone https://github.com/b45ch1/pyadolc.git
export PYTHONPATH=$SOFTWARE/PyADOL-C:$PYTHONPATH
cd pyadolc

# Setup, build, install
./bootstrap.sh
CC=gcc CXX=g++ python3 setup.py build
python3 setup.py install

# Test
python3 -m pip install matplotlib nose
python3 -c "import adolc; adolc.test()"
