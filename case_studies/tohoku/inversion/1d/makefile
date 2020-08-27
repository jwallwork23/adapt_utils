all: discrete continuous

discrete:
	@echo "Running discrete adjoint inversion on mesh 1/3..."
	@python3 discrete.py -level 0 -recompute_parameter_space 1 -rerun_optimisation 1
	@echo "Running discrete adjoint inversion on mesh 2/3..."
	@python3 discrete.py -level 1 -recompute_parameter_space 1 -rerun_optimisation 1
	@echo "Running discrete adjoint inversion on mesh 3/3..."
	@python3 discrete.py -level 2 -recompute_parameter_space 1 -rerun_optimisation 1

continuous:
	@echo "Running continuous adjoint inversion on mesh 1/3..."
	@python3 continuous.py -level 0 -rerun_optimisation 1
	@echo "Running continuous adjoint inversion on mesh 2/3..."
	@python3 continuous.py -level 1 -rerun_optimisation 1
	@echo "Running continuous adjoint inversion on mesh 3/3..."
	@python3 continuous.py -level 2 -rerun_optimisation 1
