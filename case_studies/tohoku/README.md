### Data used for Tohoku tsunami simulation

Includes:
  * Bathymetry file `resources/bathymetry.nc` provided by [GEBCO][GEBCO] (the General Bathymetric
    Chart of the Oceans).
  * Latitude-longitude coordinates of gauges 801, 802, 803, 804 and 806 provided by [PARI][PARI]
    (the Port and Airport Research Institute of Japan). 
  * Initial free surface field `resources/surf.nc`, inverted in [1] and provided by the authors.
  * Modified initial surface field `resources/surf_zeroed.nc`, adjusted for far field average using
    GMT.
  * Pressure gauge timeseries files `resources/p02.dat` and `resources/p06.dat`, provided by the
    authors of [1].
  * Latitude-longitude coordinates of gauges P02 and P06 provided by the authors of [1].

[1] T. Saito, Y. Ito, D. Inazu, & R. Hino, "Tsunami source of the 2011 Tohoku‐Oki earthquake,
    Japan: Inversion analysis based on dispersive tsunami simulations" (2011), Geophysical Research
    Letters, 38(7).

[GEBCO]: https://www.gebco.net "GEBCO"
[PARI]: https://www.pari.go.jp/en/ "PARI"
