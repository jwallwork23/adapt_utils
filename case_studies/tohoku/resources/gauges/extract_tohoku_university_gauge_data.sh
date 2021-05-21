#!/bin/bash

# Download GPS, tide and coastal wave gauge data
unzip -o p02_p06.zip
mv p02.dat P02.txt
mv p06.dat P06.txt
rm -rf __MACOSX
