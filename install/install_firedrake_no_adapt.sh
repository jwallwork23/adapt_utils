#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake without Pragmatic.                #
#                                                                        #
#                                       Joe Wallwork, 2nd September 2020 #
# ====================================================================== #

unset PYTHONPATH

# Check SOFTWARE environment variable
if [ ! -e "$SOFTWARE" ]; then
	echo "SOFTWARE environment variable $SOFTWARE does not exist."
	exit 1
fi

# Environment variables for MPI
export MPICC=/usr/bin/mpicc
export MPICXX=/usr/bin/mpicxx
export MPIF90=/usr/bin/mpif90
export MPIEXEC=/usr/bin/mpiexec

# Environment variables for Firedrake installation
export FIREDRAKE_ENV=firedrake-no-adapt
export FIREDRAKE_DIR=$SOFTWARE/$FIREDRAKE_ENV

# Check environment variables
echo "MPICC="$MPICC
echo "MPICXX="$MPICXX
echo "MPIF90="$MPIF90
echo "MPIEXEC="$MPIEXEC
echo "FIREDRAKE_ENV="$FIREDRAKE_ENV
echo "FIREDRAKE_DIR="$FIREDRAKE_DIR
echo "python3="$(which python3)
echo "Are these settings okay? Press enter to continue."
read chk

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --install thetis --venv-name $FIREDRAKE_ENV \
	--mpicc $MPICC --mpicxx $MPICXX --mpif90 $MPIF90 --mpiexec $MPIEXEC
source $FIREDRAKE_DIR/bin/activate

# Very basic test of installation
cd $FIREDRAKE_DIR/src/firedrake/demos
make
python3 helmholtz/helmholtz.py

# Install pip dependencies for adapt_utils
./install_pip_dependencies
