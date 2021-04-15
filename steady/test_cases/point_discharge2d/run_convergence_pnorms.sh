#!/bin/bash

for approach in dwr weighted_hessian weighted_gradient; do
	for i in 1 2 3 4 5 6 7 8 9 10; do
		python3 run_adaptation_loop.py \
			-family cg -stabilisation supg -approach $approach -norm_order $i;
		python3 run_adaptation_loop.py \
			-family cg -stabilisation supg -approach $approach -norm_order $i -offset 1;
	done
done
