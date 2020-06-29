all: fixed_mesh

# TODO: refine_equator mesh generator (store to dmplex)

fixed_mesh:
	@echo "Solving equatorial Rossby wave problem on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"

uniform:
	@echo "Solving equatorial Rossby wave problem on a hierarchy of fixed meshes..."
	@python3 run_fixed_mesh.py -debug 1 -n_coarse 1
	@python3 run_fixed_mesh.py -debug 1 -n_coarse 2
	@python3 run_fixed_mesh.py -debug 1 -n_coarse 4
	@echo "Done!"

refined_equator:
	@echo "Solving equatorial Rossby wave problem on a hierarchy of fixed meshes refined around the equator..."
	@python3 run_fixed_mesh.py -debug 1 -n_coarse 1 -refine_equator 1
	@python3 run_fixed_mesh.py -debug 1 -n_coarse 2 -refine_equator 1
	@python3 run_fixed_mesh.py -debug 1 -n_coarse 4 -refine_equator 1
	@echo "Done!"

moving_mesh:
	@echo "Solving equatorial Rossby wave problem on a moving mesh..."
	@python3 run_moving_mesh.py
	@echo "Done!"
