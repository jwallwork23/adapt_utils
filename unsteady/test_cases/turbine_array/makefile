all: clean mesh plot

mesh:
	@echo "Generating Delauney mesh of turbine array domain..."
	@gmsh -2 channel.geo

mesh_box:
	@echo "Generating Delauney mesh of turbine array domain..."
	@gmsh -2 channel_box.geo

spin_up:
	@echo "Spinning up hydrodynamics..."
	@python3 spin_up.py

fixed_mesh:
	@echo "Running fixed mesh simulation..."
	@python3 run_fixed_mesh.py -plot_pvd 1

hessian:
	@echo "Running Hessian-based mesh adaptive simulation..."
	@python3 run_hessian_based.py -plot_pvd 1

plot:
	@echo "Plotting mesh for turbine array problem..."
	@python3 plot_mesh.py

clean:
	@echo "Cleaning directory..."
	@rm -rf *.msh
