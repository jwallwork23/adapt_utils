#!/bin/bash

for g in 801 802 803 804 806 807 811 812 813 815 P02 P06 KPG1 KPG2 MPG1 MPG2 21401 21413 21418 21419
do
	echo "Preprocessing gauge $g..."
	./preproc.py $g
done
