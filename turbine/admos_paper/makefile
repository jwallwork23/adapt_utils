all: geo mesh_aligned mesh_offset

geo:
	@echo python3 generate_geo.py

mesh_aligned:
	gmsh -2 xcoarse_2_turbine.geo
	gmsh -2 coarse_2_turbine.geo
	gmsh -2 medium_2_turbine.geo
	gmsh -2 fine_2_turbine.geo
	gmsh -2 xfine_2_turbine.geo

mesh_offset:
	gmsh -2 xcoarse_2_offset_turbine.geo
	gmsh -2 coarse_2_offset_turbine.geo
	gmsh -2 medium_2_offset_turbine.geo
	gmsh -2 fine_2_offset_turbine.geo
	gmsh -2 xfine_2_offset_turbine.geo

clean:
	rm -Rf steady/outputs/*
