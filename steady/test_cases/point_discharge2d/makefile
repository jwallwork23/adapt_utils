all: dir uniform table

dir:
	@echo "Making directories..."
	@mkdir -p outputs/fixed_mesh/hdf5
	@mkdir -p outputs/dwr/hdf5
	@mkdir -p outputs/anisotropic_dwr/hdf5
	@mkdir -p outputs/weighted_hessian/hdf5
	@mkdir -p outputs/weighted_gradient/hdf5
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

# --- Adaptation runs based on 1st order forward result

adaptive_forward: dwr isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient

dwr:
	@echo "Running vertexwise isotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr -offset 1

isotropic_dwr:
	@echo "Running elementwise isotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr -offset 1

anisotropic_dwr:
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

# --- Adaptation runs based on 1st order adjoint result

adaptive_adjoint: dwr_adjoint isotropic_dwr_adjoint anisotropic_dwr_adjoint weighted_hessian_adjoint

dwr_adjoint:
	@echo "Running vertexwise isotropic adjoint DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr_adjoint
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr_adjoint -offset 1

isotropic_dwr_adjoint:
	@echo "Running elementwise anisotropic adjoint DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr_adjoint
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr_adjoint -offset 1

anisotropic_dwr_adjoint:
	@echo "Running anisotropic adjoint DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr_adjoint
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr_adjoint -offset 1

weighted_hessian_adjoint:
	@echo "Running adjoint 'weighted Hessian' anisotropic adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian_adjoint
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian_adjoint -offset 1

weighted_gradient_adjoint:
	@echo "Running adjoint 'weighted gradient' anisotropic adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient_adjoint
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient_adjoint -offset 1

# --- Adaptation runs based on 2nd order result / combinations of first order results

adaptive_avg: dwr_avg isotropic_dwr_avg \
	anisotropic_dwr_avg weighted_hessian_avg weighted_gradient_avg

adaptive_int: dwr_int isotropic_dwr_int \
	anisotropic_dwr_int weighted_hessian_int weighted_gradient_int

dwr_avg:
	@echo "Running vertexwise second order isotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr_avg
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr_avg -offset 1

isotropic_dwr_avg:
	@echo "Running elementwise second order isotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr_avg
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr_avg -offset 1

anisotropic_dwr_avg:
	@echo "Running averaged first order anisotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr_avg
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr_avg -offset 1

weighted_hessian_avg:
	@echo "Running averaged 'weighted Hessian' adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian_avg
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian_avg -offset 1

weighted_gradient_avg:
	@echo "Running averaged 'weighted gradient' adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient_avg
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient_avg -offset 1

dwr_int:
	@echo "Running vertexwise intersected first order isotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr_int
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach dwr_int -offset 1

isotropic_dwr_int:
	@echo "Running elementwise intersected first order anisotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr_int
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach isotropic_dwr_int -offset 1

anisotropic_dwr_int:
	@echo "Running intersected first order anisotropic DWR adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr_int
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-convergence_rate 2 -approach anisotropic_dwr_int -offset 1

weighted_hessian_int:
	@echo "Running intersected 'weighted Hessian' adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian_int
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_hessian_int -offset 1

weighted_gradient_int:
	@echo "Running intersected 'weighted gradient' adaptation convergence test..."
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient_int
	@python3 run_adaptation_loop.py -family cg -stabilisation supg -anisotropic_stabilisation 1 \
		-norm_order 1 -approach weighted_gradient_int -offset 1

# --- Convergence plots

plot: plot_by_estimator plot_by_approach

plot_by_estimator: plot_forward plot_adjoint plot_avg plot_int

plot_by_approach: plot_dwr plot_anisotropic_dwr plot_weighted_hessian plot_weighted_gradient

plot_forward:
	@echo "Plotting forward convergence curves..."
	@python3 plot_convergence.py forward -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_adjoint:
	@echo "Plotting adjoint convergence curves..."
	@python3 plot_convergence.py adjoint -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_avg:
	@echo "Plotting averaged convergence curves..."
	@python3 plot_convergence.py avg -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_int:
	@echo "Plotting intersected convergence curves..."
	@python3 plot_convergence.py int -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_dwr:
	@echo "Plotting dwr convergence curves..."
	@python3 plot_convergence.py dwr -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_anisotropic_dwr:
	@echo "Plotting anisotropic_dwr convergence curves..."
	@python3 plot_convergence.py anisotropic_dwr -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_weighted_hessian:
	@echo "Plotting weighted_hessian convergence curves..."
	@python3 plot_convergence.py weighted_hessian -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_weighted_gradient:
	@echo "Plotting weighted_gradient convergence curves..."
	@python3 plot_convergence.py weighted_gradient -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

plot_isotropic:
	@echo "Plotting isotropic convergence curves..."
	@python3 plot_convergence.py isotropic -family cg -stabilisation supg \
		-anisotropic_stabilisation 1 -norm_order 1 -convergence_rate 2

convergence: uniform table adaptive_forward plot_forward
