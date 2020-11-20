all: dir clean geo base_mesh uniform table adapt plot screenshot

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
	@cd resources/meshes && rm -Rf *.geo *.msh


# --- Initial meshes

geo:
	@echo "Generating geometry files..."
	@python3 meshgen.py 0 0
	@python3 meshgen.py 0 1
	@python3 meshgen.py 0 2
	@python3 meshgen.py 1 0
	@python3 meshgen.py 1 1
	@python3 meshgen.py 1 2
	@python3 meshgen.py 2 0
	@python3 meshgen.py 2 1
	@python3 meshgen.py 2 2
	@python3 meshgen.py 3 0
	@python3 meshgen.py 3 1
	@python3 meshgen.py 3 2
	@python3 meshgen.py 4 0
	@python3 meshgen.py 4 1
	@python3 meshgen.py 4 2

base_mesh:
	@echo "Generating base meshes..."
	@gmsh -2 resources/meshes/channel_0_0.geo
	@gmsh -2 resources/meshes/channel_0_1.geo
	@gmsh -2 resources/meshes/channel_0_2.geo

mesh:
	@echo "Generating all meshes..."
	@gmsh -2 resources/meshes/channel_0_0.geo
	@gmsh -2 resources/meshes/channel_0_1.geo
	@gmsh -2 resources/meshes/channel_0_2.geo
	@gmsh -2 resources/meshes/channel_1_0.geo
	@gmsh -2 resources/meshes/channel_1_1.geo
	@gmsh -2 resources/meshes/channel_1_2.geo
	@gmsh -2 resources/meshes/channel_2_0.geo
	@gmsh -2 resources/meshes/channel_2_1.geo
	@gmsh -2 resources/meshes/channel_2_2.geo
	@gmsh -2 resources/meshes/channel_3_0.geo
	@gmsh -2 resources/meshes/channel_3_1.geo
	@gmsh -2 resources/meshes/channel_3_2.geo
	@gmsh -2 resources/meshes/channel_4_0.geo
	@gmsh -2 resources/meshes/channel_4_1.geo
	@gmsh -2 resources/meshes/channel_4_2.geo


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
