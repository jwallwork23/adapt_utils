### Data used for Tohoku tsunami simulation

Includes:
  * Bathymetry file `resources/gebco.nc` provided by [GEBCO][GEBCO] (the General Bathymetric Chart of the Oceans).
  * Bathymetry file `resources/etopo1.nc` provided by [NOAA][NOAA] (the National Oceanic and Atmospheric Administration.
  * Timeseries for GPS gauges 801, 802, 803, 804 and 806 provided by [PARI][PARI] (the Port and Airport Research Institute of Japan). 
  * Initial free surface field `resources/surf.nc`, inverted in [1] and provided by the authors.
  * Modified initial surface field `resources/surf_zeroed.nc`, adjusted for far field average using GMT.
  * Pressure gauge timeseries files `resources/P02.dat` and `resources/P06.dat`, provided by the authors of [1].
  * (Processed versions of) timeseries for pressure gauges KPG1, KPG1, MPG1 and MPG2 provided by [JAMSTEC][JAMSTEC] (the Japanese Agency for Marine-Earth Science and Technology).
  * (Processed versions of) timeseries for [DART][DART] pressure gauges 21401, 21414, 21418 and 21419 provided by NOAA.


[1] T. Saito, Y. Ito, D. Inazu, & R. Hino, "Tsunami source of the 2011 Tohoku‐Oki earthquake,
    Japan: Inversion analysis based on dispersive tsunami simulations" (2011), Geophysical Research
    Letters, 38(7).

[GEBCO]: https://www.gebco.net "GEBCO"
[NOAA]: https://www.ngdc.noaa.gov/mgg/global "NOAA"
[PARI]: https://www.pari.go.jp/en/ "PARI"
[JAMSTEC]: http://www.jamstec.go.jp/scdc/top_e.html "JAMSTEC"
[DART]: https://www.ndbc.noaa.gov "DART"
