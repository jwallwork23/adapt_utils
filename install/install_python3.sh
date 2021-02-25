#!/bin/bash

# ====================================================================== #
# Bash script for installing Python 3.7.4., assuming the operating       #
# system is Ubuntu. (Firedrake requires Python 3.6 or later.)            #
#                                                                        #
# If running on a fresh Ubuntu OS (such as on a newly spun-up virtual    #
# machine) that doesn't have Python 3.6 or later then this script        #
# should be run first, before the `install_firedrake.sh` script.         #
#                                                                        #
# This script is particularly useful if the system has anaconda          #
# installed and the user doesn't want to permanently replace it.         #
#                                                                        #
#                                       Joe Wallwork, 3rd September 2020 #
# ====================================================================== #

# Check existence of SOFTWARE environment variable
if [ ! -e "$SOFTWARE" ]; then
    echo "SOFTWARE environment variable $SOFTWARE does not exist."
    exit 1
fi
cd $SOFTWARE

# Install dependencies
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
	libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev

# Download binary and decompress
wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
tar -xf Python-3.7.4.tgz
cd Python-3.7.4

# Configure and install
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# Alias for convenience
alias python3='python3.7'
