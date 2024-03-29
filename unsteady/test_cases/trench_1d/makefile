all: dir setup_hydrodynamics benchmark rtol freq alpha plot

dir:
	mkdir -p adapt_output
	mkdir -p fixed_output
	mkdir -p plots

setup_hydrodynamics:
	python3 trench_hydro.py -res 0.0625
	python3 trench_hydro.py -res 0.125
	python3 trench_hydro.py -res 0.25
	python3 trench_hydro.py -res 0.5
	python3 trench_hydro.py -res 1
	python3 trench_hydro.py -res 2
	python3 trench_hydro.py -res 4

res:
	mkdir -p outputs/res
	python3 run_fixed_mesh.py -res 0.0625 > outputs/res/0.0625
	python3 run_fixed_mesh.py -res 0.1250 > outputs/res/0.1250
	python3 run_fixed_mesh.py -res 0.2500 > outputs/res/0.2500
	python3 run_fixed_mesh.py -res 0.5000 > outputs/res/0.5000
	python3 run_fixed_mesh.py -res 1.0000 > outputs/res/1.0000
	python3 run_fixed_mesh.py -res 2.0000 > outputs/res/2.0000

benchmark: res
	python3 run_fixed_mesh.py -res 4 > outputs/res/4.0000

rtol: rtol_5 rtol_10 rtol_20

rtol_5:
	mkdir -p outputs/rtol/0.0625
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-1 > outputs/rtol/0.0625/1e-01
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-2 > outputs/rtol/0.0625/1e-02
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-3 > outputs/rtol/0.0625/1e-03
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-4 > outputs/rtol/0.0625/1e-04
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-5 > outputs/rtol/0.0625/1e-05
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-6 > outputs/rtol/0.0625/1e-06
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-7 > outputs/rtol/0.0625/1e-07
	python3 run_moving_mesh.py -res 0.0625 -rtol 1e-8 > outputs/rtol/0.0625/1e-08

rtol_10:
	mkdir -p outputs/rtol/0.1250
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-1 > outputs/rtol/0.1250/1e-01
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-2 > outputs/rtol/0.1250/1e-02
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-3 > outputs/rtol/0.1250/1e-03
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-4 > outputs/rtol/0.1250/1e-04
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-5 > outputs/rtol/0.1250/1e-05
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-6 > outputs/rtol/0.1250/1e-06
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-7 > outputs/rtol/0.1250/1e-07
	python3 run_moving_mesh.py -res 0.125 -rtol 1e-8 > outputs/rtol/0.1250/1e-08

rtol_20:
	mkdir -p outputs/rtol/0.2500
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-1 > outputs/rtol/0.2500/1e-01
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-2 > outputs/rtol/0.2500/1e-02
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-3 > outputs/rtol/0.2500/1e-03
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-4 > outputs/rtol/0.2500/1e-04
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-5 > outputs/rtol/0.2500/1e-05
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-6 > outputs/rtol/0.2500/1e-06
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-7 > outputs/rtol/0.2500/1e-07
	python3 run_moving_mesh.py -res 0.25 -rtol 1e-8 > outputs/rtol/0.2500/1e-08

freq: freq_5 freq_10 freq_20

freq_5:
	mkdir -p outputs/freq/0.0625
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 5 > outputs/freq/0.0625/5
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 10 > outputs/freq/0.0625/10
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 20 > outputs/freq/0.0625/20
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 40 > outputs/freq/0.0625/40
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 120 > outputs/freq/0.0625/120
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 360 > outputs/freq/0.0625/360
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 1080 > outputs/freq/0.0625/1080
	python3 run_moving_mesh.py -res 0.0625 -alpha 2 -dt_per_mesh_movement 2160 > outputs/freq/0.0625/2160
	python3 run_fixed_mesh.py -res 0.0625 > outputs/res/0.0625

freq_10:
	mkdir -p outputs/freq/0.1250
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 5 > outputs/freq/0.1250/5
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 10 > outputs/freq/0.1250/10
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 20 > outputs/freq/0.1250/20
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 40 > outputs/freq/0.1250/40
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 120 > outputs/freq/0.1250/120
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 360 > outputs/freq/0.1250/360
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 1080 > outputs/freq/0.1250/1080
	python3 run_moving_mesh.py -res 0.125 -alpha 2 -dt_per_mesh_movement 2160 > outputs/freq/0.1250/2160
	python3 run_fixed_mesh.py -res 0.125 > outputs/res/0.1250

freq_20:
	mkdir -p outputs/freq/0.2500
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 5 > outputs/freq/0.2500/5
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 10 > outputs/freq/0.2500/10
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 20 > outputs/freq/0.2500/20
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 40 > outputs/freq/0.2500/40
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 120 > outputs/freq/0.2500/120
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 360 > outputs/freq/0.2500/360
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 1080 > outputs/freq/0.2500/1080
	python3 run_moving_mesh.py -res 0.25 -alpha 2 -dt_per_mesh_movement 2160 > outputs/freq/0.2500/2160
	python3 run_fixed_mesh.py -res 0.25 > outputs/res/0.2500

alpha:
	./run_alpha.sh

plot: plot_rtol plot_freq plot_alpha plot_res

plot_rtol:
	python3 plot_rtol.py -res 0.0625
	python3 plot_rtol.py -res 0.125
	python3 plot_rtol.py -res 0.25

plot_freq:
	python3 plot_freq.py -res 0.0625
	python3 plot_freq.py -res 0.125
	python3 plot_freq.py -res 0.25

plot_alpha:
	python3 plot_alpha.py -freq 40
	python3 plot_alpha.py -freq 120
	python3 plot_alpha.py -freq 360

plot_res:
	python3 plot_res.py
