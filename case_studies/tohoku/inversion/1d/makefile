all: discrete continuous

discrete:
	@python3 discrete.py 0
	@python3 discrete.py 1
	@python3 discrete.py 2

plot_discrete:
	@python3 plot_progress.py 0 discrete
	@python3 plot_progress.py 1 discrete
	@python3 plot_progress.py 2 discrete

continuous:
	@echo "Running continuous adjoint inversion on mesh 1/3..."
	@python3 continuous.py -level 0 -rerun_optimisation 1
	@echo "Running continuous adjoint inversion on mesh 2/3..."
	@python3 continuous.py -level 1 -rerun_optimisation 1
	@echo "Running continuous adjoint inversion on mesh 3/3..."
	@python3 continuous.py -level 2 -rerun_optimisation 1
