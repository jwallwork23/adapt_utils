all: dir mesh uniform metric_based monitor_based fixed_mesh

# --- Setup

dir:
	@echo "Creating output directory..."
	@mkdir -p outputs/meshes
	@echo "Done!"

mesh:
	@echo "Generating meshes..."
	@python3 meshgen.py
	@echo "Done!"
	@echo "Post-processing meshes..."
	@python3 resources/meshes/postproc.py -dm_plex_prescribed_boundary_labels 100,200,300 -dm_plex_prescribed_boundary_sizes 100e3,10e3,5e3 -dm_plex_remesh_bd
	@echo "Done!"

# Plotting

# TODO: Update
uniform:
	@echo "Plotting bathymetry, initial surface and Coriolis parameter on uniform mesh..."
	@python3 plot_uniform.py
	@echo "Done!"

# TODO: Update
metric_based:
	@echo "Plotting bathymetry and initial surface on mesh adapted to bathymetry Hessian..."
	@python3 plot_metric_based.py
	@echo "Done!"

# TODO: Update
monitor_based:
	@echo "Plotting bathymetry and initial surface on mesh adapted to bathymetry Hessian..."
	@python3 plot_monitor_based.py
	@echo "Done!"

# --- Convergence analysis

qmesh_linear:
	@echo "Running convergence analysis using linear shallow water equations..."
	@python3 run_qmesh_convergence.py
	@echo "Done!"

qmesh_nonlinear:
	@echo "Running convergence analysis using nonlinear shallow water equations..."
	@python3 run_qmesh_convergence.py -nonlinear 1
	@echo "Done!"

qmesh: qmesh_linear qmesh_nonlinear

# --- Run scripts

fixed_mesh:
	@echo "Solving Tohoku tsunami problem on a fixed mesh..."
	@python3 run_fixed_mesh.py
	@echo "Done!"
#
#  NOTE: We specify different gradation parameters for different boundary segments:
#    * Artificial open ocean (tag 100): 100 km;
#    * Coast outside of Fukushima (tag 200): 10 km;
#    * Coast around Fukushima (tag 300): 5 km.
#
#  NOTE: The -dm_plex_remesh_bd option can deform the domain boundary, which is why we specify a
#    gradation for the open ocean boundary. Instead of applying it here, we use it in the mesh
#    postprocessing (see `mesh` above).

# TODO: Hessian based

dwp:
	@echo "Solving Tohoku tsunami problem using isotropic 'dual weighted primal' adaptation..."
	@python3 run_dwp.py -dm_plex_prescribed_boundary_labels 200,300 -dm_plex_prescribed_boundary_sizes 10e3,5e3
	@echo "Done!"
# @python3 run_dwp.py -dm_plex_prescribed_boundary_labels 100,200,300 -dm_plex_prescribed_boundary_sizes 100e3,10e3,5e3

dwr:
	@echo "Solving Tohoku tsunami problem using isotropic 'dual weighted residual' adaptation..."
	@python3 run_dwr.py -dm_plex_prescribed_boundary_labels 200,300 -dm_plex_prescribed_boundary_sizes 10e3,5e3
	@echo "Done!"

# Cleanup

clean:
	@echo "Cleaning directory..."
	@rm -rf outputs/*.pvd
	@rm -rf outputs/*.vtu
	@rm -rf outputs/fixed_mesh/*.pvd
	@rm -rf outputs/fixed_mesh/*.vtu
	@rm -rf outputs/fixed_mesh/Elevation2d/*.pvd
	@rm -rf outputs/fixed_mesh/Elevation2d/*.vtu
	@rm -rf outputs/fixed_mesh/Velocity2d/*.pvd
	@rm -rf outputs/fixed_mesh/Velocity2d/*.vtu
	@echo "Done!"
