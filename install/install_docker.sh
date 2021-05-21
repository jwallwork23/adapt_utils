#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake using the docker image found on   #
# DockerHub.                                                             #
#                                                                        #
# If Docker is not set up then you will need to run `setup_docker.sh`    #
# first.                                                                 #
#                                                                        #
#                                      Joe Wallwork, 11th September 2020 #
# ====================================================================== #

# Pull Firedrake docker image
docker pull firedrakeproject/firedrake

# Create an instance
docker run -v $HOME:$HOME --rm -it firedrakeproject/firedrake
