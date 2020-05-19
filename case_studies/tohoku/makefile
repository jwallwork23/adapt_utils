all: dir mesh uniform metric_based monitor_based fixed_mesh

dir:
	@echo "Creating output directory..."
	@mkdir -p outputs/meshes
	@echo "Done!"

mesh:
	@echo "Generating meshes..."
	@python3 meshgen.py 0
	@python3 meshgen.py 1
	@python3 meshgen.py 2
	@python3 meshgen.py 3
	@python3 meshgen.py 4
	@echo "Done!"

uniform:
	@echo "Plotting bathymetry, initial surface and Coriolis parameter on uniform mesh..."
	@python3 plot_uniform.py
	@echo "Done!"

metric_based:
	@echo "Plotting bathymetry and initial surface on mesh adapted to bathymetry Hessian..."
	@python3 plot_metric_based.py
	@echo "Done!"

monitor_based:
	@echo "Plotting bathymetry and initial surface on mesh adapted to bathymetry Hessian..."
	@python3 plot_monitor_based.py
	@echo "Done!"

fixed_mesh:
	@echo "Solving Tohoku tsunami problem on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"

# TODO: Hessian based

dwp:
	@echo "Solving Tohoku tsunami problem using 'dual weighted primal' adaptation..."
	@python3 run_dwp.py -dm_plex_prescribed_boundary_labels 200 -dm_plex_prescribed_boundary_sizes 10e3
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -rf outputs/*.pvd
	@rm -rf outputs/*.vtu
	@rm -rf outputs/fixed_mesh/*.pvd
	@rm -rf outputs/fixed_mesh/*.vtu
	@rm -rf outputs/fixed_mesh/Elevation2d/*.pvd
	@rm -rf outputs/fixed_mesh/Elevation2d/*.vtu
	@rm -rf outputs/fixed_mesh/Velocity2d/*.pvd
	@rm -rf outputs/fixed_mesh/Velocity2d/*.vtu
	@echo "Done!"
