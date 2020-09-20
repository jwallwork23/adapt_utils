all: discrete continuous

discrete:
	@echo "Running discrete adjoint inversion on mesh 1/3..."
	@python3 invert.py discrete -level 0 -recompute_parameter_space 1 -rerun_optimisation 1
	@echo "Running discrete adjoint inversion on mesh 2/3..."
	@python3 invert.py discrete -level 1 -recompute_parameter_space 1 -rerun_optimisation 1
	@echo "Running discrete adjoint inversion on mesh 3/3..."
	@python3 invert.py discrete -level 2 -recompute_parameter_space 1 -rerun_optimisation 1

plot_discrete:
	@echo "Plotting discrete adjoint inversion outputs on mesh 1/3..."
	@python3 plot_convergence.py discrete -level 0 -plot_all 1
	@echo "Plotting discrete adjoint inversion outputs on mesh 2/3..."
	@python3 plot_convergence.py discrete -level 1 -plot_all 1
	@echo "Plotting discrete adjoint inversion outputs on mesh 3/3..."
	@python3 plot_convergence.py discrete -level 2 -plot_all 1

continuous:
	@echo "Running continuous adjoint inversion on mesh 1/3..."
	@python3 continuous.py -level 0 -rerun_optimisation 1
	@echo "Running continuous adjoint inversion on mesh 2/3..."
	@python3 continuous.py -level 1 -rerun_optimisation 1
	@echo "Running continuous adjoint inversion on mesh 3/3..."
	@python3 continuous.py -level 2 -rerun_optimisation 1
