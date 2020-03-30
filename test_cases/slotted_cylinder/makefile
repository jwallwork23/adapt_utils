all: clean mesh

circle: uniform_circle plot_circle print_circle

square: uniform_square plot_square print_square

mesh:
	@echo "Generating Delauney mesh of circular domain..."
	@gmsh -2 circle.geo
	@echo "Done!"

uniform: uniform_circle uniform_square

uniform_circle:
	@echo "Solving solid body rotation problem on circular domain..."
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 0
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 1
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 2
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 3
	@echo "Done!"

uniform_square:
	@echo "Solving solid body rotation problem on square domain..."
	@python3 run_thetis.py -approach fixed_mesh -mesh_type square -init_res 0
	@python3 run_thetis.py -approach fixed_mesh -mesh_type square -init_res 1
	@python3 run_thetis.py -approach fixed_mesh -mesh_type square -init_res 2
	@python3 run_thetis.py -approach fixed_mesh -mesh_type square -init_res 3
	@echo "Done!"

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
	@echo "Done!"
	@gio open outputs/fixed_mesh/convergence_circle.pdf

plot_square:
	@echo "Plotting convergence for square domain..."
	@python3 plot_convergence.py -mesh_type square
	@echo "Done!"
	@gio open outputs/fixed_mesh/convergence_square.pdf

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
	@echo "Done!"
