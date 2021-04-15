<<<<<<< HEAD
all: mesh

mesh:
	gmsh -2 spaceship.geo

plot:
	python3 plot_setup.py
=======
all: clean mesh plot

mesh:
	@echo "Generating Delauney mesh of 'spaceship' domain..."
	@gmsh -2 spaceship.geo

plot:
	@echo "Plotting setup of 'spaceship' problem..."
	@python3 plot_setup.py
	@python3 plot_tidal_forcing.py

clean:
	@echo "Cleaning directory..."
	@rm -Rf *.msh
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
