all: clean geo_xcoarse mesh_xcoarse

allmesh: clean geo mesh_aligned mesh_offset

geo:
	@echo "Generating geometry files..."
	@python3 generate_geo.py
	@echo "Done!"

mesh_aligned:
	@echo "Generating aligned meshes..."
	@gmsh -2 xcoarse.geo
	@gmsh -2 coarse.geo
	@gmsh -2 medium.geo
	@gmsh -2 fine.geo
	@gmsh -2 xfine.geo
	@echo "Done!"

mesh_offset:
	@echo "Generating aligned meshes..."
	@gmsh -2 xcoarse_offset.geo
	@gmsh -2 coarse_offset.geo
	@gmsh -2 medium_offset.geo
	@gmsh -2 fine_offset.geo
	@gmsh -2 xfine_offset.geo
	@echo "Done!"

geo_xcoarse:
	@echo "Generating xcoarse geometry files..."
	@python3 generate_geo.py -level xcoarse
	@echo "Done!"

mesh_xcoarse:
	@echo "Generating xcoarse meshes..."
	@gmsh -2 xcoarse.geo
	@gmsh -2 xcoarse_offset.geo
	@echo "Done!"

uniform:
	@echo "Running uniform convergence test..."
	@echo mpiexec -n 8 python3 run_uniform_convergence.py
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.geo *.msh
	@echo "Done!"
