all: clean dir mesh mesh_postproc gauge


# --- Setup

dir:
	@echo "Creating output directory..."
	@mkdir -p outputs
	@echo "Done!"

mesh:
	@echo "Generating meshes..."
	@mkdir -p resources/meshes
	@python meshgen.py
	@echo "Done!"

mesh_postproc:
	@echo "Post-processing meshes..."
	@python3 resources/meshes/postproc.py -dm_plex_prescribed_boundary_labels 100,200,300 -dm_plex_prescribed_boundary_sizes 100e3,10e3,5e3 -dm_plex_remesh_bd
	@echo "Done!"


# --- Gauges

gauge:
	@echo "Plotting gauge data..."
	@python3 plot_gauge_data.py
	@echo "Done!"


# --- Cleanup

clean:
	@echo "Cleaning directory..."
	@rm -rf resources/meshes/*.msh
	@rm -rf resources/meshes/*.h5
	@echo "Done!"
