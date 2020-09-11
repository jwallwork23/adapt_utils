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
# The Firedrake virtual environment should be activated and the          #
# $FIREDRAKE_DIR environment variable must be set.                       #
#                                                                        #
#                                      Joe Wallwork, 11th September 2020 #
# ====================================================================== #


# Install pip dependencies for adapt_utils
python3 -m pip install matplotlib netCDF4 numpy pandas scipy utide
# python3 -m pip install jupyter
# python3 -m pip install qmesh  # FIXME

# Set environment variables
if [ ! -e "$FIREDRAKE_DIR" ]; then
    echo "FIREDRAKE_DIR environment variable $FIREDRAKE_DIR does not exist."
    exit 1
fi

# Install ClawPack
export CLAW_SRC=$FIREDRAKE_DIR/clawpack_src
python3 -m pip install --src=$CLAW_SRC -e git+https://github.com/clawpack/clawpack.git@v5.7.0#egg=clawpack-v5.7.0
export CLAW=$CLAW_SRC/clawpack-v5-7-0

# Install PyADOL-C
# TODO
