all: plot

run:
	python3 discrete_wave_eqn.py 2 southern_pressure -uniform_slip 1 -uniform_rake 90
	python3 discrete_wave_eqn.py 2 mid_field_pressure -uniform_slip 1 -uniform_rake 90
	python3 discrete_wave_eqn.py 2 far_field_pressure -uniform_slip 1 -uniform_rake 90
	python3 discrete_wave_eqn.py 2 all -uniform_slip 1 -uniform_rake 90 -normalise 1

plot:
	python3 view_wave_eqn.py 2 -category south
	python3 view_wave_eqn.py 2 -category mid
	python3 view_wave_eqn.py 2 -category far
	python3 view_wave_eqn.py 2 -category all
