all: clean mesh


# --- Mesh

mesh:
	@echo "Generating Delauney mesh of circular domain..."
	@gmsh -2 circle.geo


# --- Uniform convergence

uniform: uniform_circle uniform_square

uniform_circle:
	@echo "Solving solid body rotation problem on circular domain..."
	@python3 run_fixed_mesh.py -geometry circle -level 0
	@python3 run_fixed_mesh.py -geometry circle -level 1
	@python3 run_fixed_mesh.py -geometry circle -level 2
	@python3 run_fixed_mesh.py -geometry circle -level 3
	@python3 run_fixed_mesh.py -geometry circle -level 4
	@python3 run_fixed_mesh.py -geometry circle -level 5

uniform_square:
	@echo "Solving solid body rotation problem on square domain..."
	@python3 run_fixed_mesh.py -geometry square -level 0
	@python3 run_fixed_mesh.py -geometry square -level 1
	@python3 run_fixed_mesh.py -geometry square -level 2
	@python3 run_fixed_mesh.py -geometry square -level 3
	@python3 run_fixed_mesh.py -geometry square -level 4
	@python3 run_fixed_mesh.py -geometry square -level 5

print_uniform_circle:
	@cat outputs/fixed_mesh/circle_1.log
	@cat outputs/fixed_mesh/circle_2.log
	@cat outputs/fixed_mesh/circle_4.log
	@cat outputs/fixed_mesh/circle_8.log
	@cat outputs/fixed_mesh/circle_16.log
	@cat outputs/fixed_mesh/circle_32.log

print_uniform_square:
	@cat outputs/fixed_mesh/square_1.log
	@cat outputs/fixed_mesh/square_2.log
	@cat outputs/fixed_mesh/square_4.log
	@cat outputs/fixed_mesh/square_8.log
	@cat outputs/fixed_mesh/square_16.log
	@cat outputs/fixed_mesh/square_32.log

plot_uniform: plot_uniform_circle plot_uniform_square

plot_uniform_circle:
	@echo "Plotting convergence for circular domain..."
	@python3 plot_convergence.py -geometry circle
	@gio open outputs/fixed_mesh/convergence_circle.pdf

plot_uniform_square:
	@echo "Plotting convergence for square domain..."
	@python3 plot_convergence.py -geometry square
	@gio open outputs/fixed_mesh/convergence_square.pdf


# --- Lagrangian convergence

lagrangian: lagrangian_circle lagrangian_square

lagrangian_circle:
	@echo "Solving solid body rotation problem on circular domain..."
	@python3 run_lagrangian.py -geometry circle -level 0
	@python3 run_lagrangian.py -geometry circle -level 1
	@python3 run_lagrangian.py -geometry circle -level 2
	@python3 run_lagrangian.py -geometry circle -level 3
	@python3 run_lagrangian.py -geometry circle -level 4
	@python3 run_lagrangian.py -geometry circle -level 5

lagrangian_square:
	@echo "Solving solid body rotation problem on square domain..."
	@python3 run_lagrangian.py -geometry square -level 0
	@python3 run_lagrangian.py -geometry square -level 1
	@python3 run_lagrangian.py -geometry square -level 2
	@python3 run_lagrangian.py -geometry square -level 3
	@python3 run_lagrangian.py -geometry square -level 4
	@python3 run_lagrangian.py -geometry square -level 5

print_lagrangian_circle:
	@cat outputs/lagrangian/circle_1.log
	@cat outputs/lagrangian/circle_2.log
	@cat outputs/lagrangian/circle_4.log
	@cat outputs/lagrangian/circle_8.log
	@cat outputs/lagrangian/circle_16.log
	@cat outputs/lagrangian/circle_32.log

print_lagrangian_square:
	@cat outputs/lagrangian/square_1.log
	@cat outputs/lagrangian/square_2.log
	@cat outputs/lagrangian/square_4.log
	@cat outputs/lagrangian/square_8.log
	@cat outputs/lagrangian/square_16.log
	@cat outputs/lagrangian/square_32.log

plot_lagrangian: plot_lagrangian_circle plot_lagrangian_square

plot_lagrangian_circle:
	@echo "Plotting Lagrangian convergence for circular domain..."
	@python3 plot_convergence.py -geometry circle -approach lagrangian
	@gio open outputs/lagrangian/convergence_circle.pdf

plot_lagrangian_square:
	@echo "Plotting Lagrangian convergence for square domain..."
	@python3 plot_convergence.py -geometry square -approach lagrangian
	@gio open outputs/lagrangian/convergence_square.pdf


# --- Cleanup

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
