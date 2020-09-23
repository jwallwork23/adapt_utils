#!/bin/bash

# ====================================================================== #
# Bash script for installing compilers and other misc. packages needed   #
# by PETSc and Firedrake, assuming the operating system is Ubuntu.       #
#                                                                        #
# If running on a fresh Ubuntu OS (such as on a newly spun-up virtual    #
# machine) then this script should be run first, before the              #
# `install_petsc.sh` or `install_firedrake.sh` scripts.                  #
#                                                                        #
#                                       Joe Wallwork, 2nd September 2020 #
# ====================================================================== #

# Update apt
sudo apt update -y

# Install compilers
sudo apt install -y gcc gfortran gxx
sudo apt install -y make cmake
sudo apt install -y mpich

# Install other required packages
sudo apt install -y zlib1g zlib1g-dev
sudo apt install -y libblas-dev liblapack-dev

# Environment variables for MPI
export MPICC=/usr/bin/mpicc
export MPICXX=/usr/bin/mpicxx
export MPIEXEC=/usr/bin/mpiexec
export MPIF90=/usr/bin/mpif90
