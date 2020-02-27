all: xcoarse uniform adapt plot screenshot

xcoarse: dir clean geo_xcoarse mesh_xcoarse

allmesh: dir clean geo mesh

adapt: isotropic anisotropic

screenshot: screenshot_uniform screenshot_adapt

screenshot_adapt: screenshot_isotropic screenshot_anisotropic

plot: semilog loglog

dir:
	@echo "Making directories..."
	@mkdir -p outputs/fixed_mesh/hdf5
	@mkdir -p outputs/carpio_isotropic/hdf5
	@mkdir -p outputs/carpio/hdf5
	@mkdir -p data
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
	@python3 run_uniform_convergence.py
	@echo "Done!"

isotropic:
	@echo "Running isotropic adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach carpio_isotropic
	@echo "Done!"

anisotropic:
	@echo "Running anisotropic adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach carpio
	@echo "Done!"

screenshot_uniform:
	@echo "Individual run for uniform refinement screenshots..."
	@python3 run.py -approach fixed_mesh -offset 0 -level 3
	@python3 run.py -approach fixed_mesh -offset 1 -level 3
	@echo "Done!"

screenshot_isotropic:
	@echo "Individual run for isotropic adaptation screenshots..."
	@python3 run.py -approach carpio_isotropic -offset 0
	@python3 run.py -approach carpio_isotropic -offset 1
	@echo "Done!"

screenshot_anisotropic:
	@echo "Individual run for anisotropic adaptation screenshots..."
	@python3 run.py -approach carpio -offset 0
	@python3 run.py -approach carpio -offset 1
	@echo "Done!"

semilog:
	@echo "Plotting convergence curves..."
	@python3 plot_convergence.py
	@echo "Done!"

loglog:
	@echo "Plotting convergence curves..."
	@python3 plot_convergence.py -loglog 1
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.geo *.msh
	@echo "Done!"
