all: clean fixed_mesh

fixed_mesh:
	@echo "Solving trench test on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"

moving_mesh:
	@echo "Solving trench test on a moving mesh..."
	@python3 run_moving_mesh.py
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh outputs*
	@git checkout hydrodynamics_trench/*.h5
	@echo "Done!"
