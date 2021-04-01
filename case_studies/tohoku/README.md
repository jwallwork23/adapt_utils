# Tohoku tsunami case study

This directory includes post-processed gauge timeseries `resources/gauges/*.dat`:
  * 8XY provided by [PARI][PARI] (the Port and Airport Research Institute of Japan). 
  * P0X provided by the authors of [1].
  * XPGY provided by [JAMSTEC][JAMSTEC] (the Japanese Agency for Marine-Earth Science and Technology).
  * 214XY provided by [NOAA][NOAA].
Bash scripts for manually downloading and processing the gauge data can be found in `resources/gauges/`.


It also includes coastal boundary data extracted from the [ETOPO1][ETOPO1] data set in `resources/boundaries/`.
The bathymetry data should be downloaded again and saved as `resources/bathymetry/etopo1.nc`.


[1] T. Saito, Y. Ito, D. Inazu, & R. Hino, "Tsunami source of the 2011 Tohoku‐Oki earthquake,
    Japan: Inversion analysis based on dispersive tsunami simulations" (2011), Geophysical Research
    Letters, 38(7).

## Additional software dependencies

  * ClawPack - see `$ADAPT_UTILS_HOME/install/install_pip_dependencies.sh`.
  * PyADOL-C - see `$ADAPT_UTILS_HOME/install/install_pyadolc.sh`.
  * WIN - see http://wwweic.eri.u-tokyo.ac.jp/WIN/pub/win/.
  * qmesh - see http://qmesh.org/download.html.

[NOAA]: https://www.ngdc.noaa.gov/mgg/global "NOAA"
[PARI]: https://www.pari.go.jp/en/ "PARI"
[JAMSTEC]: http://www.jamstec.go.jp/scdc/top_e.html "JAMSTEC"
[DART]: https://www.ndbc.noaa.gov "DART"
[ETOPO1]: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ "ETOPO1"
