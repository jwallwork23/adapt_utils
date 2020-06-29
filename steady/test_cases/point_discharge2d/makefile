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
	@echo "Done!"

uniform:
	@echo "Running uniform convergence test..."
	@python3 run_uniform_convergence.py
	@echo "Done!"

table:
	@echo "Writing to LaTeX formatted table..."
	@python3 write_table.py
	@echo "Done!"
