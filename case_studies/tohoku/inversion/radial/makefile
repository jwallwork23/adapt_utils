all: project

project:
	@echo "Testing projection for radial basis"
	@python3 test_project.py -plot_all 1

discrete_synthetic:
	@echo "Running discrete adjoint inversion in piecewise constant basis on mesh 1/3..."
	@python3 discrete.py -level 0 -rerun_optimisation 1 -plot_all 1
	@echo "Running discrete adjoint inversion in piecewise constant basison mesh 2/3..."
	@python3 discrete.py -level 1 -rerun_optimisation 1 -plot_all 1
	@echo "Running discrete adjoint inversion in piecewise constant basis on mesh 3/3..."
	@python3 discrete.py -level 2 -rerun_optimisation 1 -plot_all 1

discrete_realistic:
	@echo "Running discrete adjoint inversion in piecewise constant basis on mesh 1/3..."
	@python3 discrete.py -level 0 -rerun_optimisation 1 -plot_all 1 -real_data 1
	@echo "Running discrete adjoint inversion in piecewise constant basison mesh 2/3..."
	@python3 discrete.py -level 1 -rerun_optimisation 1 -plot_all 1 -real_data 1
	@echo "Running discrete adjoint inversion in piecewise constant basis on mesh 3/3..."
	@python3 discrete.py -level 2 -rerun_optimisation 1 -plot_all 1 -real_data 1
