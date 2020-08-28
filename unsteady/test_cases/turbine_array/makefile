all: clean mesh plot

mesh:
	@echo "Generating Delauney mesh of turbine array domain..."
	@gmsh -2 channel.geo

spin_up:
	@echo "Spinning up hydrodynamics..."
	@python3 spin_up.py

plot:
	@echo "Plotting mesh for turbine array problem..."
	@python3 plot_mesh.py

clean:
	@echo "Cleaning directory..."
	@rm -rf *.msh
