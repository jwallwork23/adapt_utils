all: discrete continuous

discrete:
	@echo "Running discrete adjoint inversion on mesh 1/3..."
	@python3 discrete.py -level 0
	@echo "Done!"
	@echo "Running discrete adjoint inversion on mesh 2/3..."
	@python3 discrete.py -level 1
	@echo "Done!"
	@echo "Running discrete adjoint inversion on mesh 3/3..."
	@python3 discrete.py -level 2
	@echo "Done!"

continuous:
	@echo "Running continuous adjoint inversion on mesh 1/3..."
	@python3 continuous.py -level 0
	@echo "Done!"
	@echo "Running continuous adjoint inversion on mesh 2/3..."
	@python3 continuous.py -level 1
	@echo "Done!"
	@echo "Running continuous adjoint inversion on mesh 3/3..."
	@python3 continuous.py -level 2
	@echo "Done!"
