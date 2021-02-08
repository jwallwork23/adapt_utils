all: uniform intersect integrate

uniform:
	python3 run_uniform_convergence.py

intersect:
	python3 run_adaptation_loop.py -hessian_time_combination intersect -levels 4

integrate:
	python3 run_adaptation_loop.py -hessian_time_combination integrate -levels 4
