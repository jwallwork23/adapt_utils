#!/bin/bash

for freq in 40 120 360; do
	for res in 0.0625 0.125 0.25 0.5; do
		for alpha in {0..15}; do
			python3 run_moving_mesh.py -res $res -alpha $alpha -dt_per_mesh_movement $freq
		done
	done
done
