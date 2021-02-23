all: run plot plot_progress

run:
	python3 discrete_wave_eqn.py 2 southern_pressure -uniform_slip 1 -uniform_rake 90
	python3 discrete_wave_eqn.py 2 mid_field_pressure -uniform_slip 1 -uniform_rake 90
	python3 discrete_wave_eqn.py 2 far_field_pressure -uniform_slip 1 -uniform_rake 90
	python3 discrete_wave_eqn.py 2 all -uniform_slip 1 -uniform_rake 90 -normalise 1

plot:
	python3 view_wave_eqn.py 2 -categories southern_pressure
	python3 view_wave_eqn.py 2 -categories mid_field_pressure
	python3 view_wave_eqn.py 2 -categories far_field_pressure
	python3 view_wave_eqn.py 2 -categories all

plot_progress:
	python3 plot_progress.py 2 -categories southern_pressure
	python3 plot_progress.py 2 -categories mid_field_pressure
	python3 plot_progress.py 2 -categories far_field_pressure
	python3 plot_progress.py 2 -categories all
