all: dir uniform table

dir:
	@echo "Making directories..."
	@mkdir -p outputs/fixed_mesh/hdf5
	@mkdir -p outputs/dwr/hdf5
	@mkdir -p outputs/dwr_averaged/hdf5
	@mkdir -p outputs/dwr_superposed/hdf5
	@mkdir -p outputs/power/hdf5
	@mkdir -p outputs/power_averaged/hdf5
	@mkdir -p outputs/power_superposed/hdf5
	@mkdir -p outputs/loseille/hdf5
	@mkdir -p outputs/loseille_averaged/hdf5
	@mkdir -p outputs/loseille_superposed/hdf5
	@mkdir -p data
	@mkdir -p plots


# --- Calibration

calibrate: calibrate_cg calibrate_dg

calibrate_cg:
	@echo "Calibrating radius for CG (no stabilisation)..."
	@python3 calibrate_radius.py 5 cg
	@echo "Calibrating radius for CG (SU stabilisation)..."
	@python3 calibrate_radius.py 5 cg -stabilisation su
	@echo "Calibrating radius for CG (SUPG stabilisation)..."
	@python3 calibrate_radius.py 5 cg -stabilisation supg

calibrate_dg:
	@echo "Calibrating radius for DG (no stabilisation)..."
	@python3 calibrate_radius.py 5 dg
	@echo "Calibrating radius for DG (Lax-Friedrichs stabilisation)..."
	@python3 calibrate_radius.py 5 dg -stabilisation lax_friedrichs

print_calibration:
	@cat outputs/fixed_mesh/cg/log
	@cat outputs/fixed_mesh/su/log
	@cat outputs/fixed_mesh/supg/log
	@cat outputs/fixed_mesh/dg/log
	@cat outputs/fixed_mesh/lax_friedrichs/log


# --- Convergence analysis

uniform:
	@echo "Running uniform refinement convergence test..."
	@python3 run_uniform_convergence.py

isotropic:
	@echo "Running isotropic adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -offset 1

# TODO: A posteriori anisotropic
# TODO: A priori anisotropic

table:
	@echo "Writing to LaTeX formatted table..."
	@python3 write_table.py
