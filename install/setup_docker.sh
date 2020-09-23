#!/bin/bash

sudo groupadd docker
sudo usermod -aG docker ${USER}

# Log out and log back in
exit 0
