all: clean mesh plot

mesh:
	@echo "Generating Delauney mesh of 'spaceship' domain..."
	@gmsh -2 spaceship.geo
	@echo "Done!"

plot:
	@echo "Plotting setup of 'spaceship' problem..."
	@python3 plot_setup.py
	@python3 plot_tidal_forcing.py
	@echo "Done!"

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
	@echo "Done!"
