all: clean mesh

mesh:
	@echo "Generating Delauney mesh of circular domain..."
	@gmsh -2 circle.geo
	@echo "Done!"

uniform_circle:
	@echo "Solving solid body rotation problem on a hierarchy of fixed meshes..."
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 0
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 1
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 2
	@python3 run_thetis.py -approach fixed_mesh -mesh_type circle -init_res 3
	@echo "Done!"

uniform_square:
	@echo "Solving solid body rotation problem on a hierarchy of fixed meshes..."
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

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
	@echo "Done!"
