all: dir clean mesh

dir:
	@echo "Creating directories..."
	@mkdir -p plots
	@mkdir -p resources

mesh:
	@echo "Generating Delauney meshes..."
	@python3 meshgen.py 0
	@cd resources && gmsh -2 channel_0.geo
	@python3 meshgen.py 1
	@cd resources && gmsh -2 channel_1.geo
	@python3 meshgen.py 2
	@cd resources && gmsh -2 channel_2.geo

clean:
	@echo "Cleaning resource directory..."
	@cd resources && rm -rf *.msh
