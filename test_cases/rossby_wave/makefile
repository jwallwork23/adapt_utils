all: fixed_mesh

# TODO: refine_equator mesh generator (store to dmplex)

fixed_mesh:
	@echo "Solving equatorial Rossby wave problem on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"

moving_mesh:
	@echo "Solving equatorial Rossby wave problem on a moving mesh..."
	@python3 run_moving_mesh.py
	@echo "Done!"
