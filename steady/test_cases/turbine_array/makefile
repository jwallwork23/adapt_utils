all: base uniform table adapt plot screenshot

base: dir clean base_geo base_mesh

box: box_geo box_mesh

adapt: isotropic anisotropic weighted_hessian

screenshot: screenshot_uniform screenshot_adapt

screenshot_adapt: screenshot_isotropic screenshot_anisotropic screenshot_weighted_hessian

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

base_geo:
	@echo "Generating base geometry files..."
	@python3 meshgen.py 0 0
	@python3 meshgen.py 0 1
	@python3 meshgen.py 0 2

box_geo:
	@echo "Generating refined box geometry files..."
	@python3 meshgen.py 0 0 -dx_refined 1
	@python3 meshgen.py 1 0 -dx_refined 1
	@python3 meshgen.py 2 0 -dx_refined 1
	@python3 meshgen.py 3 0 -dx_refined 1
	@python3 meshgen.py 4 0 -dx_refined 1

geo: base_geo
	@echo "Generating other geometry files..."
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

box_mesh:
	@echo "Generating refined box meshes..."
	@gmsh -2 resources/meshes/channel_refined_0.geo
	@gmsh -2 resources/meshes/channel_refined_1.geo
	@gmsh -2 resources/meshes/channel_refined_2.geo
	@gmsh -2 resources/meshes/channel_refined_3.geo
	@gmsh -2 resources/meshes/channel_refined_4.geo

mesh: base_mesh
	@echo "Generating other meshes..."
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
	@python3 run_adaptive_convergence.py -approach isotropic_dwr

anisotropic:
	@echo "Running anisotropic adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach anisotropic_dwr

weighted_hessian:
	@echo "Running WH adaptive convergence test..."
	@python3 run_adaptive_convergence.py -approach weighted_hessian


# --- Screenshots

screenshot_uniform:
	@echo "Individual run for uniform refinement screenshots..."
	@python3 run_fixed_mesh.py -offset 0 -level 0 -plot_png 1
	@python3 run_fixed_mesh.py -offset 1 -level 0 -plot_png 1
	@python3 run_fixed_mesh.py -offset 0 -level 3 -plot_png 1
	@python3 run_fixed_mesh.py -offset 1 -level 3 -plot_png 1

screenshot_isotropic:
	@echo "Individual run for isotropic adaptation screenshots..."
	@python3 run_adapt.py isotropic_dwr -offset 0 -plot_png 1
	@python3 run_adapt.py isotropic_dwr -offset 1 -plot_png 1

screenshot_anisotropic:
	@echo "Individual run for anisotropic adaptation screenshots..."
	@python3 run_adapt.py anisotropic_dwr -offset 0 -plot_png 1
	@python3 run_adapt.py anisotropic_dwr -offset 1 -plot_png 1

screenshot_weighted_hessian:
	@echo "Individual run for WH adaptation screenshots..."
	@python3 run_adapt.py weighted_hessian -offset 0 -plot_png 1
	@python3 run_adapt.py weighted_hessian -offset 1 -plot_png 1


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
