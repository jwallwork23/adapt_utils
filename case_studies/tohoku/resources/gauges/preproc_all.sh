#!/bin/bash

# for g in 801 802 803 804 806 807 811 812 813 815 P02 P06
for g in KPG1 KPG2 MPG1 MPG2
do
	echo $g
	./preproc.py $g
done
