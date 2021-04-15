all: dir uniform table

dir:
	@echo "Making directories..."
	@mkdir -p outputs/fixed_mesh/hdf5
	@mkdir -p outputs/dwr/hdf5
	@# mkdir -p outputs/dwr_averaged/hdf5
	@# mkdir -p outputs/dwr_superposed/hdf5
	@mkdir -p outputs/anisotropic_dwr/hdf5
	@# mkdir -p outputs/anisotropic_dwr_averaged/hdf5
	@# mkdir -p outputs/anisotropic_dwr_superposed/hdf5
	@mkdir -p outputs/weighted_hessian/hdf5
	@# mkdir -p outputs/weighted_hessian_averaged/hdf5
	@# mkdir -p outputs/weighted_hessian_superposed/hdf5
	@mkdir -p outputs/weighted_gradient/hdf5
	@# mkdir -p outputs/weighted_gradient_averaged/hdf5
	@# mkdir -p outputs/weighted_gradient_superposed/hdf5
	@mkdir -p data
	@mkdir -p plots


# --- Calibration

calibrate: calibrate_cg calibrate_dg

calibrate_cg:
	@echo "Calibrating radius for CG (no stabilisation)..."
	@python3 calibrate_radius.py 4 cg
	@echo "Calibrating radius for CG (isotropic SU stabilisation)..."
	@python3 calibrate_radius.py 4 cg -stabilisation su
	@echo "Calibrating radius for CG (isotropic SUPG stabilisation)..."
	@python3 calibrate_radius.py 4 cg -stabilisation supg
	@echo "Calibrating radius for CG (anisotropic SU stabilisation)..."
	@python3 calibrate_radius.py 4 cg -stabilisation su -anisotropic_stabilisation 1
	@echo "Calibrating radius for CG (anisotropic SUPG stabilisation)..."
	@python3 calibrate_radius.py 4 cg -stabilisation supg -anisotropic_stabilisation 1

calibrate_dg:
	@echo "Calibrating radius for DG (no stabilisation)..."
	@python3 calibrate_radius.py 4 dg
	@echo "Calibrating radius for DG (isotropic Lax-Friedrichs stabilisation)..."
	@python3 calibrate_radius.py 4 dg -stabilisation lax_friedrichs
	@echo "Calibrating radius for DG (anisotropic Lax-Friedrichs stabilisation)..."
	@python3 calibrate_radius.py 4 dg -stabilisation lax_friedrichs -anisotropic_stabilisation 1

print_calibration:
	@cat outputs/fixed_mesh/cg/log
	@cat outputs/fixed_mesh/su/log
	@cat outputs/fixed_mesh/supg/log
	@cat outputs/fixed_mesh/dg/log
	@cat outputs/fixed_mesh/lax_friedrichs/log


# --- Convergence analysis

uniform:
	@echo "Running uniform refinement convergence test..."
	@python3 run_uniform_convergence.py cg -stabilisation supg -anisotropic_stabilisation 1

table:
	@echo "Writing to LaTeX formatted table..."
	@python3 write_table.py

adaptive: isotropic anisotropic weighted_hessian weighted_gradient

isotropic:
	@echo "Running isotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr -offset 1

anisotropic:
	@echo "Running anisotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr -offset 1

weighted_hessian:
	@echo "Running 'weighted Hessian' anisotropic adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian -offset 1

weighted_gradient:
	@echo "Running 'weighted gradient' anisotropic adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient -offset 1

plot:
	@echo "Plotting convergence curves..."
	@python3 plot_convergence.py cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -convergence_rate 2

plot_norms:
	@echo "Plotting convergence curves across norm orders..."
	@python3 plot_convergence_norms.py dwr cg -stabilisation supg
	@python3 plot_convergence_norms.py weighted_hessian cg -stabilisation supg
	@python3 plot_convergence_norms.py weighted_gradient cg -stabilisation supg
	@python3 plot_convergence_norms.py anisotropic_dwr cg -stabilisation supg

convergence: uniform table adaptive plot
