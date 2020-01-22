all: clean mesh

mesh:
	@echo "Generating mesh for inundated beach..."
	@gmsh -2 strip.geo
	@echo "Done!"

fixed_mesh:
	@echo "Solving inundated beach problem on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"

moving_mesh:
	@echo "Solving inundated beach problem on a moving mesh..."
	@python3 run_moving_mesh.py
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
	@echo "Done!"
