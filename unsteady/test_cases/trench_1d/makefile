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

benchmark:
	python3 run_fixed_mesh.py -res 4 > outputs/res/4.0000

rtol: rtol_40 rtol_80

rtol_40:
	mkdir -p outputs/rtol/0.5000
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-1 > outputs/rtol/0.5000/1e-01
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-2 > outputs/rtol/0.5000/1e-02
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-3 > outputs/rtol/0.5000/1e-03
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-4 > outputs/rtol/0.5000/1e-04
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-5 > outputs/rtol/0.5000/1e-05
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-6 > outputs/rtol/0.5000/1e-06
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-7 > outputs/rtol/0.5000/1e-07
	python3 run_moving_mesh.py -res 0.5 -rtol 1e-8 > outputs/rtol/0.5000/1e-08

rtol_80:
	mkdir -p outputs/rtol/1.0000
	python3 run_moving_mesh.py -res 1 -rtol 1e-1 > outputs/rtol/1.0000/1e-01
	python3 run_moving_mesh.py -res 1 -rtol 1e-2 > outputs/rtol/1.0000/1e-02
	python3 run_moving_mesh.py -res 1 -rtol 1e-3 > outputs/rtol/1.0000/1e-03
	python3 run_moving_mesh.py -res 1 -rtol 1e-4 > outputs/rtol/1.0000/1e-04
	python3 run_moving_mesh.py -res 1 -rtol 1e-5 > outputs/rtol/1.0000/1e-05
	python3 run_moving_mesh.py -res 1 -rtol 1e-6 > outputs/rtol/1.0000/1e-06
	python3 run_moving_mesh.py -res 1 -rtol 1e-7 > outputs/rtol/1.0000/1e-07
	python3 run_moving_mesh.py -res 1 -rtol 1e-8 > outputs/rtol/1.0000/1e-08

freq:
	mkdir -p outputs/freq
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 5 > outputs/freq/5
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 10 > outputs/freq/10
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 20 > outputs/freq/20
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 40 > outputs/freq/40
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 120 > outputs/freq/120
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 360 > outputs/freq/360
	python3 run_moving_mesh.py -res 0.5 -alpha 2 -dt_per_mesh_movement 1080 > outputs/freq/1080

res:
	mkdir -p outputs/res
	python3 run_fixed_mesh.py -res 0.0625 > outputs/res/0.0625
	# python3 run_fixed_mesh.py -res 0.1000 > outputs/res/0.1000
	python3 run_fixed_mesh.py -res 0.1250 > outputs/res/0.1250
	python3 run_fixed_mesh.py -res 0.2500 > outputs/res/0.2500
	# python3 run_fixed_mesh.py -res 0.3750 > outputs/res/0.3750
	python3 run_fixed_mesh.py -res 0.5000 > outputs/res/0.5000
	# python3 run_fixed_mesh.py -res 0.7500 > outputs/res/0.7500
	python3 run_fixed_mesh.py -res 1.0000 > outputs/res/1.0000
	python3 run_fixed_mesh.py -res 2.0000 > outputs/res/2.0000