all: setup_hydrodynamics rtol res

setup_hydrodynamics:
	python3 trench_hydro.py -res 0.0625
	# python3 trench_hydro.py -res 0.1
	python3 trench_hydro.py -res 0.125
	python3 trench_hydro.py -res 0.25
	# python3 trench_hydro.py -res 0.375
	python3 trench_hydro.py -res 0.5
	# python3 trench_hydro.py -res 0.75
	python3 trench_hydro.py -res 1
	python3 trench_hydro.py -res 2
	python3 trench_hydro.py -res 4

rtol:
	mkdir -p outputs/rtol
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-1 > outputs/rtol/1e-1
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-2 > outputs/rtol/1e-2
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-3 > outputs/rtol/1e-3
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-4 > outputs/rtol/1e-4
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-5 > outputs/rtol/1e-5
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-6 > outputs/rtol/1e-6
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-7 > outputs/rtol/1e-7
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-8 > outputs/rtol/1e-8

res:
	mkdir -p outputs/res
	python3 run_moving_mesh.py -res 0.0625 > outputs/res/0.0625
	# python3 run_moving_mesh.py -res 0.1 > outputs/res/0.1
	python3 run_moving_mesh.py -res 0.125 > outputs/res/0.125
	python3 run_moving_mesh.py -res 0.25 > outputs/res/0.25
	# python3 run_moving_mesh.py -res 0.375 > outputs/res/0.375
	python3 run_moving_mesh.py -res 0.5 > outputs/res/0.5
	# python3 run_moving_mesh.py -res 0.75 > outputs/res/0.75
	python3 run_moving_mesh.py -res 1 > outputs/res/1
	python3 run_moving_mesh.py -res 2 > outputs/res/2
	python3 run_moving_mesh.py -res 4 > outputs/res/4
