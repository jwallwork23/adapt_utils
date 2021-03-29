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

# KNOWN INSTALL BUG:
#   You will need to remove the NULL return statement and then comment out these lines
echo "Remove the 'return NULL;' satement in $VIRTUAL_ENV/lib/python$(python3 -c "print('$(python3 --version)'[-5:-2])")/site-packages/numpy/core/include/numpy/__multiarray_api.h."
exit

# Install dependencies
sudo apt-get update -y
sudo apt-get install -y libtool libboost-all-dev wget

# Clone repo from GitHub
cp pyadolc_python3.patch $SOFTWARE/
cd $SOFTWARE
if [[ ! -f PyADOL-C ]]; then

	# Download PyADOL-C
	mkdir PyADOL-C
	cd PyADOL-C
	git clone https://github.com/b45ch1/pyadolc.git
	export PYTHONPATH=$SOFTWARE/PyADOL-C:$PYTHONPATH
	cd pyadolc

	# Install ADOL-C
	./bootstrap.sh

	# Apply compile flags patch for Python3
	git apply $SOFTWARE/pyadolc_python3.patch
	cd ../..
fi
cd PyADOL-C/pyadolc

# Build
#    You might need to:
#        cd /usr/lib/x86_64-linux-gnu/
#        sudo ln -s libboost_python3x.so libboost_python3.so
#        sudo ln -s libboost_numpy3x.so libboost_numpy3.so
#    where 3x stands for the version 3.x
CC=gcc CXX=g++ python3 setup.py build

# Install
#   If you don't have write access to python3.x/site-packages then you will need to specify --prefix
python3 setup.py install

# Test (outside of install directory)
cd $SOTWARE
python3 -c "import adolc; adolc.test()"
