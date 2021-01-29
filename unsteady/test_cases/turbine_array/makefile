all: clean mesh plot

mesh:
	@echo "Generating Delauney meshes of turbine array domain..."
	@mkdir -p resources
	@python3 meshgen.py 0
	@cd resources && gmsh -2 channel_box_0.geo
	@python3 meshgen.py 1
	@cd resources && gmsh -2 channel_box_1.geo
	@python3 meshgen.py 2
	@cd resources && gmsh -2 channel_box_2.geo
	@python3 meshgen.py 3
	@cd resources && gmsh -2 channel_box_3.geo
	@python3 meshgen.py 4
	@cd resources && gmsh -2 channel_box_4.geo
	@python3 meshgen.py 5
	@cd resources && gmsh -2 channel_box_5.geo

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
	@echo "Cleaning resource directory..."
	@cd resources && rm -rf *.msh
