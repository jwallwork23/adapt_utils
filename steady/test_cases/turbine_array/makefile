all: xcoarse uniform table adapt plot screenshot

xcoarse: dir clean geo_xcoarse mesh_xcoarse

allmesh: dir clean geo mesh

adapt: isotropic anisotropic

screenshot: screenshot_uniform screenshot_adapt

screenshot_adapt: screenshot_isotropic screenshot_anisotropic

plot: semilog loglog


# --- Misc

dir:
	@echo "Making directories..."
	@mkdir -p outputs/fixed_mesh/hdf5
	@mkdir -p outputs/carpio_isotropic/hdf5
	@mkdir -p outputs/carpio/hdf5
	@mkdir -p data
	@mkdir -p resources/meshes

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.geo *.msh


# --- Initial meshes

geo:
	@echo "Generating geometry files..."
	@python3 generate_geo.py

mesh:
	@echo "Generating all meshes..."
	@gmsh -2 resources/meshes/xcoarse_0.geo
	@gmsh -2 resources/meshes/coarse_0.geo
	@gmsh -2 resources/meshes/medium_0.geo
	@gmsh -2 resources/meshes/fine_0.geo
	@gmsh -2 resources/meshes/xfine_0.geo
	@gmsh -2 resources/meshes/xcoarse_1.geo
	@gmsh -2 resources/meshes/coarse_1.geo
	@gmsh -2 resources/meshes/medium_1.geo
	@gmsh -2 resources/meshes/fine_1.geo
	@gmsh -2 resources/meshes/xfine_1.geo
	@gmsh -2 resources/meshes/xcoarse_2.geo
	@gmsh -2 resources/meshes/coarse_2.geo
	@gmsh -2 resources/meshes/medium_2.geo
	@gmsh -2 resources/meshes/fine_2.geo
	@gmsh -2 resources/meshes/xfine_2.geo

geo_xcoarse:
	@echo "Generating xcoarse geometry files..."
	@python3 generate_geo.py -level xcoarse

mesh_xcoarse:
	@echo "Generating xcoarse meshes..."
	@gmsh -2 resources/meshes/xcoarse_0.geo
	@gmsh -2 resources/meshes/xcoarse_1.geo
	@gmsh -2 resources/meshes/xcoarse_2.geo


# --- Convergence runs

uniform:
	@echo "Running uniform convergence test..."
	@python3 run_uniform_convergence.py -offset 0
	@python3 run_uniform_convergence.py -offset 1

isotropic:
	@echo "Running isotropic adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach carpio_isotropic

anisotropic:
	@echo "Running anisotropic adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach carpio


# --- Screenshots

screenshot_uniform:
	@echo "Individual run for uniform refinement screenshots..."
	@python3 run_fixed_mesh.py -approach fixed_mesh -offset 0 -level 3
	@python3 run_fixed_mesh.py -approach fixed_mesh -offset 1 -level 3

screenshot_isotropic:
	@echo "Individual run for isotropic adaptation screenshots..."
	@python3 run_adapt.py -approach carpio_isotropic -offset 0
	@python3 run_adapt.py -approach carpio_isotropic -offset 1

screenshot_anisotropic:
	@echo "Individual run for anisotropic adaptation screenshots..."
	@python3 run_adapt.py -approach carpio -offset 0
	@python3 run_adapt.py -approach carpio -offset 1


# --- Convergence plots

semilog:
	@echo "Plotting convergence curves..."
	@python3 plot_convergence.py

loglog:
	@echo "Plotting convergence curves..."
	@python3 plot_convergence.py -loglog 1

table:
	@echo "Writing to LaTeX formatted table..."
	@python3 write_table.py
