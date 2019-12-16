all: clean geo mesh_aligned mesh_offset

xcoarse: clean geo_xcoarse mesh_xcoarse

geo:
	@echo "Generating geometry files..."
	@python3 generate_geo.py

mesh_aligned:
	@echo "Generating aligned meshes..."
	@gmsh -2 xcoarse.geo
	@gmsh -2 coarse.geo
	@gmsh -2 medium.geo
	@gmsh -2 fine.geo
	@gmsh -2 xfine.geo

mesh_offset:
	@echo "Generating aligned meshes..."
	@gmsh -2 xcoarse_offset.geo
	@gmsh -2 coarse_offset.geo
	@gmsh -2 medium_offset.geo
	@gmsh -2 fine_offset.geo
	@gmsh -2 xfine_offset.geo

geo_xcoarse:
	@echo "Generating xcoarse geometry files..."
	@python3 generate_geo.py -level xcoarse

mesh_xcoarse:
	@echo "Generating xcoarse meshes..."
	@gmsh -2 xcoarse.geo
	@gmsh -2 xcoarse_offset.geo

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.geo *.msh
