all: dir mesh gauges clean


dir:
	@echo "Creating output directory..."
	@mkdir -p outputs
	@echo "Done!"


# --- Meshes

mesh:
	@echo "Generating meshes..."
	@mkdir -p resources/meshes
	@python meshgen.py
	@echo "Done!"

mesh_postproc:
	@echo "Post-processing meshes..."
	@python3 resources/meshes/postproc.py \
		-dm_plex_prescribed_boundary_labels 100,200,300 \
		-dm_plex_prescribed_boundary_sizes 100e3,10e3,5e3 -dm_plex_remesh_bd
	@echo "Done!"


# --- Gauges

manual_download_gauges:
	@echo "Checking gauge data which require manual downloads exist..."
	@python3 resources/gauges/check_manual_downloads.py
	@echo "Done!"

download_gauges:
	@echo "Downloading gauge data..."
	@cd resources/gauges/ && ./extract_gps_gauge_data.sh
	@echo "Done!"

extract_gauges:
	@echo "Extracting gauge data..."
	@cd resources/gauges/ && ./extract_kpg.sh && ./extract_mpg.sh
	@echo "Done!"

preproc_gauges:
	@echo "Pre-processing gauge data..."
	@python3 resources/gauges/preproc.py all
	@echo "Done!"

plot_gauges:
	@echo "Plotting gauge data..."
	@python3 plot_gauge_data.py
	@echo "Done!"

clean_gauges:
	@echo "Cleaning gauge data directory..."
	@cd resources/gauges && rm -rf outputs/ && rm -rf __pycache__/ && rm *_mmh.*
	@echo "Done!"

gauges: download_gauges extract_gauges preproc_gauges plot_gauges clean_gauges


# --- Cleanup

clean: clean_gauges
