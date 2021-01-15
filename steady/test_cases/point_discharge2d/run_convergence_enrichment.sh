#!/bin/bash

for approach in dwr isotropic_dwr anisotropic_dwr; do
	for method in GE_hp GE_h GE_p DQ; do
		python3 run_adaptation_loop.py -family cg -stabilisation supg \
			-anisotropic_stabilisation 1 -norm_order 1 \
			-approach $approach -enrichment_method $method
		python3 run_adaptation_loop.py -family cg -stabilisation supg \
			-anisotropic_stabilisation 1 -norm_order 1 -offset 1 \
			-approach $approach -enrichment_method $method
	done
done
