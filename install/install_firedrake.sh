#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a PETSc installation     #
# which uses Pragmatic.                                                  #
#                                                                        #
# The `install_petsc.sh` script should be run first.                     #
#                                                                        #
# Note that we use the following custom branches:                        #
#   * firedrake: joe/meshadapt                                           #
#   * petsc4py:  joe/dm-adapt-cell-tags                                  #
#                                                                        #
# Most of the modifications were made by Nicolas Barral. Minor updates   #
# by Joe Wallwork.                                                       #
# ====================================================================== #

unset PYTHONPATH

# Environment variables for MPI
export MPICC=/usr/bin/mpicc
export MPICXX=/usr/bin/mpicxx
export MPIEXEC=/usr/bin/mpiexec
export MPIF90=/usr/bin/mpif90
for mpi in mpicc mpicxx mpiexec mpif90; do
	if [ ! -f /usr/bin/$mpi ]; then
		echo "Cannot find $mpi in /usr/bin."
		exit 1
	fi
done

# Environment variables for Firedrake installation
export FIREDRAKE_ENV=firedrake-adapt
export FIREDRAKE_DIR=$SOFTWARE/$FIREDRAKE_ENV

# Check environment variables
echo "MPICC="$MPICC
echo "MPICXX="$MPICXX
echo "MPIF90="$MPIF90
echo "MPIEXEC="$MPIEXEC
echo "PETSC_DIR="$PETSC_DIR
if [ ! -e "$PETSC_DIR" ]; then
    echo "$PETSC_DIR does not exist. Please run install_petsc.sh."
    exit 1
fi
echo "PETSC_ARCH="$PETSC_ARCH
echo "FIREDRAKE_ENV="$FIREDRAKE_ENV
echo "FIREDRAKE_DIR="$FIREDRAKE_DIR
echo "python3="$(which python3)
echo "Are these settings okay? Press enter to continue."
read chk

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --honour-petsc-dir --install thetis --venv-name $FIREDRAKE_ENV \
	--mpicc $MPICC --mpicxx $MPICXX --mpif90 $MPIF90 --mpiexec $MPIEXEC \
	--package-branch petsc4py joe/dm-adapt-cell-tags \
	--package-branch firedrake joe/meshadapt_patched
        # --package-branch firedrake joe/meshadapt
source $FIREDRAKE_DIR/bin/activate

# Very basic test of installation
cd $FIREDRAKE_DIR/src/firedrake
python3 tests/test_adapt_2d.py

# Install pip dependencies for adapt_utils
python3 -m pip install matplotlib netCDF4 numpy pandas scipy utide
# python3 -m pip install jupyter qmesh
