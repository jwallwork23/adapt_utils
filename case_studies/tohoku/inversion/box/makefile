all: project discrete

project:
	@echo "Testing projection for piecewise constant basis"
	@python3 test_project.py -plot_all 1

discrete:
	@echo "Running discrete adjoint inversion in  piecewise constant basis on mesh 1/3..."
	@python3 discrete.py -level 0 -rerun_optimisation 1 -plot_all 1
	@echo "Running discrete adjoint inversion in  piecewise constant basison mesh 2/3..."
	@python3 discrete.py -level 1 -rerun_optimisation 1 -plot_all 1
	@echo "Running discrete adjoint inversion in  piecewise constant basis on mesh 3/3..."
	@python3 discrete.py -level 2 -rerun_optimisation 1 -plot_all 1
