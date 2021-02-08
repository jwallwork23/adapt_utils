#!/bin/bash

# ====================================================================== #
# Bash script for installing dependencies of adapt_utils which may be    #
# obtained via pip.                                                      #
#                                                                        #
# Basic dependencies:                                                    #
#  * matplotlib                                                          #
#  * netCDF4                                                             #
#  * numpy                                                               #
#  * pandas                                                              #
#  * scipy                                                               #
#  * utide                                                               #
#                                                                        #
# Dependencies for tsunami modelling:                                    #
#  * ClawPack                                                            #
#  * PyADOL-C                                                            #
#                                                                        #
# The virtual environment should be activated.                           #
#                                                                        #
#                                      Joe Wallwork, 11th September 2020 #
# ====================================================================== #


# Check virtual environment is active
if [ ! -e "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is not active."
    exit 1
fi

# Install pip dependencies for adapt_utils
python3 -m pip install matplotlib netCDF4 numpy pandas scipy utide
# python3 -m pip install jupyter
# python3 -m pip install qmesh  # FIXME

# Install ClawPack
export CLAW_SRC=$VIRTUAL_ENV/src/clawpack_src
python3 -m pip install --src=$CLAW_SRC -e git+https://github.com/clawpack/clawpack.git@v5.7.0#egg=clawpack-v5.7.0
export CLAW=$CLAW_SRC/clawpack-v5-7-0

# Basic test of installation
if [ ! -e "$CLAW" ]; then
    echo "CLAW environment variable $CLAW does not exist."
    exit 1
fi
python3 -c "import clawpack"
