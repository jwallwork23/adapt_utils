#!/bin/bash

# ====================================================================== #
# Bash script for plotting Tohoku tsunami source inversion experimental  #
# results saved in some output directory. Pass this as an extension      #
# <ext>, under which the output directory becomes `realistic_<ext>`.     #
#                                                                        #
# Plots include:                                                         #
#   (a) progress of both the (continuous) QoI and its gradient during    #
#       the optimisation routine;                                        #
#   (b) convergence of both continuous and discrete forms of the QoI;    #
#   (c) convergence of the mean square error between timeseries;         #
#   (d) optimised source fields.                                         #
#                                                                        #
#                                      Joe Wallwork, 11th September 2020 #
# ====================================================================== #

ext=$1

# Plot QoI convergence and compute other errors
for basis in box radial; do
	python3 plot_convergence.py $basis -plot_all 1 -plot_initial_guess 1 -extension $ext
done

# Compare convergence curves
python3 plot_convergence_all.py box,radial -plot_all 1 -extension $ext

# Plot optimised sources
for basis in box radial; do
	for level in 0 1 2; do
		python3 plot_optimised_source.py $basis -plot_all 1 -level $level -extension $ext
	done
done
