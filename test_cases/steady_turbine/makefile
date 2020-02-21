all: dir clean geo_xcoarse mesh_xcoarse

allmesh: dir clean geo mesh

results: all uniform adapt plot

dir:
	@echo "Making directories..."
	@mkdir -p outputs/fixed_mesh/hdf5
	@mkdir -p outputs/carpio_isotropic/hdf5
	@mkdir -p outputs/carpio/hdf5
	@echo "Done!"

geo:
	@echo "Generating geometry files..."
	@python3 generate_geo.py
	@echo "Done!"

mesh:
	@echo "Generating all meshes..."
	@gmsh -2 xcoarse_0.geo
	@gmsh -2 coarse_0.geo
	@gmsh -2 medium_0.geo
	@gmsh -2 fine_0.geo
	@gmsh -2 xfine_0.geo
	@gmsh -2 xcoarse_1.geo
	@gmsh -2 coarse_1.geo
	@gmsh -2 medium_1.geo
	@gmsh -2 fine_1.geo
	@gmsh -2 xfine_1.geo
	@gmsh -2 xcoarse_2.geo
	@gmsh -2 coarse_2.geo
	@gmsh -2 medium_2.geo
	@gmsh -2 fine_2.geo
	@gmsh -2 xfine_2.geo
	@echo "Done!"

geo_xcoarse:
	@echo "Generating xcoarse geometry files..."
	@python3 generate_geo.py -level xcoarse
	@echo "Done!"

mesh_xcoarse:
	@echo "Generating xcoarse meshes..."
	@gmsh -2 xcoarse_0.geo
	@gmsh -2 xcoarse_1.geo
	@gmsh -2 xcoarse_2.geo
	@echo "Done!"

uniform:
	@echo "Running uniform convergence test..."
	@mpiexec -n 8 python3 run_uniform_convergence.py
	@echo "Done!"

adapt:
	@echo "Running adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach carpio_isotropic
	@python3 run_adaptive_convergence.py -approach carpio
	@echo "Done!"

plot:
	@echo "Plotting convergence curves..."
	@python3 plot_convergence.py
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.geo *.msh
	@echo "Done!"
