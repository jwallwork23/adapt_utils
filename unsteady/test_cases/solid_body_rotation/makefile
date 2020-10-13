all: clean mesh


# --- Mesh

circle: uniform_circle plot_circle print_circle

square: uniform_square plot_square print_square

mesh:
	@echo "Generating Delauney mesh of circular domain..."
	@gmsh -2 circle.geo


# --- Uniform convergence

uniform: uniform_circle uniform_square

uniform_circle:
	@echo "Solving solid body rotation problem on circular domain..."
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type circle -level 0
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type circle -level 1
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type circle -level 2
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type circle -level 3

uniform_square:
	@echo "Solving solid body rotation problem on square domain..."
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type square -level 0
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type square -level 1
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type square -level 2
	@python3 run_fixed_mesh.py -approach fixed_mesh -mesh_type square -level 3

print_circle:
	@cat outputs/fixed_mesh/circle_1.log
	@cat outputs/fixed_mesh/circle_2.log
	@cat outputs/fixed_mesh/circle_4.log
	@cat outputs/fixed_mesh/circle_8.log

print_square:
	@cat outputs/fixed_mesh/square_1.log
	@cat outputs/fixed_mesh/square_2.log
	@cat outputs/fixed_mesh/square_4.log
	@cat outputs/fixed_mesh/square_8.log

plot: plot_circle plot_square

plot_circle:
	@echo "Plotting convergence for circular domain..."
	@python3 plot_convergence.py -mesh_type circle
	@gio open outputs/fixed_mesh/convergence_circle.pdf

plot_square:
	@echo "Plotting convergence for square domain..."
	@python3 plot_convergence.py -mesh_type square
	@gio open outputs/fixed_mesh/convergence_square.pdf


# --- Cleanup

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
