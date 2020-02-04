all: dir uniform bathymetry_hessian fixed_mesh

dir:
	@echo "Creating output directory..."
	@mkdir outputs
	@echo "Done!"

uniform:
	@echo "Plotting bathymetry, initial surface and Coriolis parameter on uniform mesh..."
	@python3 plot_uniform.py
	@echo "Done!"

bathymetry_hessian:
	@echo "Plotting bathymetry and initial surface on mesh adapted to bathymetry Hessian..."
	@python3 plot_bathymetry_hessian.py
	@echo "Done!"

fixed_mesh:
	@echo "Solving Tohoku tsunami problem on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"
