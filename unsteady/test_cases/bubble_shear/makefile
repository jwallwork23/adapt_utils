all: uniform intersect integrate on_the_fly metric_advection

uniform:
	python3 run_uniform_convergence.py

intersect:
	python3 run_adaptation_loop.py -hessian_time_combination intersect -max_level 4

integrate:
	python3 run_adaptation_loop.py -hessian_time_combination integrate -max_level 4

on_the_fly:
	python3 run_on_the_fly.py 0
	python3 run_on_the_fly.py 1
	python3 run_on_the_fly.py 2
	python3 run_on_the_fly.py 3

metric_advection:
	python3 run_on_the_fly.py 0 -metric_advection 1
	python3 run_on_the_fly.py 1 -metric_advection 1
	python3 run_on_the_fly.py 2 -metric_advection 1
	python3 run_on_the_fly.py 3 -metric_advection 1
