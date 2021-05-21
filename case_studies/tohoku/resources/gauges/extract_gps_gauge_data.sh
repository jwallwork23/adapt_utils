#!/bin/bash

# Download GPS, tide and coastal wave gauge data
wget https://nowphas.mlit.go.jp/pastdatars/data/NOWPHAS_Tsunami_data.zip
unzip NOWPHAS_Tsunami_data.zip
mv NOWPHAS_Tsunami_data GPS_gauges
rm NOWPHAS_Tsunami_data.zip

# Copy GPS data into current directory
cp GPS_gauges/2011TET801G.txt 801.txt
cp GPS_gauges/2011TET802G.txt 802.txt
cp GPS_gauges/2011TET803G.txt 803.txt
cp GPS_gauges/2011TET804G.txt 804.txt
cp GPS_gauges/2011TET806G.txt 806.txt
cp GPS_gauges/2011TET807G.txt 807.txt
cp GPS_gauges/2011TET811G.txt 811.txt
cp GPS_gauges/2011TET812G.txt 812.txt
cp GPS_gauges/2011TET813G.txt 813.txt
cp GPS_gauges/2011TET815G.txt 815.txt

# Download document with information on gauges
wget https://nowphas.mlit.go.jp/pastdatars/PDF/list/dai_2017p.pdf
mv dai_2017p.pdf GPS_gauge_info.pdf
