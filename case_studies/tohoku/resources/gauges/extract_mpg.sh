#!/bin/bash

# ====================================================================== #
# Bash script for extracting elevation timeseries data from the WIN      #
# format used at the Japan Agency for Marine-Earth Science and           #
# Technology (JAMSTEC). Script is hard-coded for the Tohoku tsunami,     #
# which occured on 2011-03-11. However, it can be easily modified for    #
# other events.                                                          #
#                                                                        #
# Data is accessible from http://www.jamstec.go.jp/scdc/top_e.html.      #
#                                                                        #
# WIN can be downloaded at http://wwweic.eri.u-tokyo.ac.jp/WIN/pub/win/. #
# ====================================================================== #

tar -xzvf muroto1655.tar.gz

# Loop over gauges KPG1 and KPG2
for i in 1 2; do

	# Clear output file
        out="MPG$i.txt"
	if [ -f $out ]; then
		echo "Overwriting old log file $out"
		rm $out
	fi
	touch $out

        # Loop over hours
	for h in {14..18}; do

		# Loop over minutes (padded with zeroes for 2 digits)
		for m in {00..59}; do
			# Restrict to the 2 hour window from 14:45 to 16:45
			if [ $h -eq 14 ] && [ $m -lt 45 ]; then
				continue
			elif [ $h -eq 18 ] && [ $m -gt 44 ]; then
				break
			fi

			# Extract data and append to output file
			#   NOTE that we need to choose the
			#   appropriate channels, CD1B and CD1D.
			fname="mpg$i"
			fname+="_mmh.20110311_"
			fname+=$h
			fname+=$m
			fname+=00
			if [ ! -f $fname ]; then
				echo "$fname does not exist! Aborting now."
				exit
			fi
			echo $fname
			if [ $i -eq 1 ]; then
				dewin -acen CD1B $fname >> $out
			else
				dewin -acen CD1D $fname >> $out
			fi
			rm $fname
		done
	done
done