#!/bin/bash

# ====================================================================== #
# Bash script for setting up docker.                                     #
#                                      Joe Wallwork, 11th September 2020 #
# ====================================================================== #

sudo groupadd docker
sudo usermod -aG docker ${USER}

# Log out and log back in
exit 0
