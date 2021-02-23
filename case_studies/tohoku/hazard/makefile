all: dir


# --- Setup

dir:
	@echo "Creating output directory..."
	@mkdir -p outputs


# --- Convergence analysis

qmesh_linear:
	@echo "Running convergence analysis using linear shallow water equations..."
	@python3 run_qmesh_convergence.py

qmesh_nonlinear:
	@echo "Running convergence analysis using nonlinear shallow water equations..."
	@python3 run_qmesh_convergence.py -nonlinear 1

qmesh: qmesh_linear qmesh_nonlinear


# --- Run scripts

fixed_mesh:
	@echo "Solving Tohoku tsunami problem on a fixed mesh..."
	@python3 run_fixed_mesh.py -plot_pvd 1

#  NOTE: There are three boundary segments, on which we can specify gradation parameters (in km):
#    * Artificial open ocean (tag 100);
#    * Coast outside of Fukushima (tag 200);
#    * Coast around Fukushima (tag 300).
#   The -dm_plex_remesh_bd option can also be used, but note that it deforms the domain boundary.

hessian:
	@echo "Solving Tohoku tsunami problem using anisotropic 'Hessian based' adaptation..."
	@python3 run_adapt.py -approach hessian -plot_pvd 1 \
		-dm_plex_prescribed_boundary_labels 200,300 \
		-dm_plex_prescribed_boundary_sizes 10e3,5e3

dwp:
	@echo "Solving Tohoku tsunami problem using isotropic 'dual weighted primal' adaptation..."
	@python3 run_adapt.py -approach dwp -plot_pvd 1 \
		-dm_plex_prescribed_boundary_labels 200,300 \
		-dm_plex_prescribed_boundary_sizes 10e3,5e3

dwr:
	@echo "Solving Tohoku tsunami problem using isotropic 'dual weighted residual' adaptation..."
	@python3 run_adapt.py -approach dwr -plot_pvd 1 \
		-dm_plex_prescribed_boundary_labels 200,300 \
		-dm_plex_prescribed_boundary_sizes 10e3,5e3

weighted_hessian:
	@echo "Solving Tohoku tsunami problem using anisotropic 'weighted Hessian' adaptation..."
	@python3 run_adapt.py -approach weighted_hessian -plot_pvd 1 \
		-dm_plex_prescribed_boundary_labels 200,300 \
		-dm_plex_prescribed_boundary_sizes 10e3,5e3


# --- Plotting

kernel: ball gaussian

ball:
	@python3 plot_kernel.py -level 3 -kernel_shape ball
	@python3 plot_kernel.py -level 3 -locations Ogasawara -radius 50e3 -kernel_shape ball

gaussian:
	@python3 plot_kernel.py -level 3
	@python3 plot_kernel.py -level 3 -locations Ogasawara -radius 50e3


# --- Cleanup

clean:
	@echo "Cleaning directory..."
	@rm -rf outputs/*.pvd
	@rm -rf outputs/*.vtu
	@rm -rf outputs/fixed_mesh/*.pvd
	@rm -rf outputs/fixed_mesh/*.vtu
	@rm -rf outputs/hessian/*.pvd
	@rm -rf outputs/hessian/*.vtu
	@rm -rf outputs/dwp/*.pvd
	@rm -rf outputs/dwp/*.vtu
	@rm -rf outputs/dwr/*.pvd
	@rm -rf outputs/weighted_hessian/*.vtu
	@rm -rf outputs/weighted_hessian/*.pvd
	@rm -rf outputs/dwr/*.vtu
