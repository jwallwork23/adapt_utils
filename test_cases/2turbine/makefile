all: clean geo mesh_aligned mesh_offset

geo:
	@echo "Generating geometry files..."
	@python3 generate_geo.py

mesh_aligned:
	@echo "Generating aligned meshes..."
	@gmsh -2 xcoarse_2_turbine.geo
	@gmsh -2 coarse_2_turbine.geo
	@gmsh -2 medium_2_turbine.geo
	@gmsh -2 fine_2_turbine.geo
	@gmsh -2 xfine_2_turbine.geo

mesh_offset:
	@echo "Generating aligned meshes..."
	@gmsh -2 xcoarse_2_offset_turbine.geo
	@gmsh -2 coarse_2_offset_turbine.geo
	@gmsh -2 medium_2_offset_turbine.geo
	@gmsh -2 fine_2_offset_turbine.geo
	@gmsh -2 xfine_2_offset_turbine.geo

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.geo *.msh
