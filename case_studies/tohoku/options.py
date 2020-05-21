from thetis import *
from thetis.configuration import *

import os
import netCDF4
import matplotlib.pyplot as plt

from adapt_utils.swe.tsunami.options import TsunamiOptions
from adapt_utils.swe.tsunami.conversion import from_latlon


__all__ = ["TohokuOptions"]


class TohokuOptions(TsunamiOptions):
    """
    Setup for model of the Tohoku tsunami which struck the east coast of Japan in 2011, leading to
    the meltdown of Daiichi nuclear power plant, Fukushima.

    Data sources:
      * Bathymetry data extracted from GEBCO.
      * Initial free surface elevation field generated by inversion on tide gauge data by
        [Saito et al.].

    [Saito et al.] T. Saito, Y. Ito, D. Inazu, R. Hino, "Tsunami source of the 2011 Tohoku‐Oki
                   earthquake, Japan: Inversion analysis based on dispersive tsunami simulations",
                   Geophysical Research Letters (2011), 38(7).
    """
    def __init__(self, mesh=None, level=0, locations=["Fukushima Daiichi", ], radii=[50.0e+03, ], **kwargs):
        self.force_zone_number = 54
        super(TohokuOptions, self).__init__(**kwargs)

        # Stabilisation
        self.use_automatic_sipg_parameter = False
        self.sipg_parameter = None
        self.base_viscosity = 0.0
        # self.base_viscosity = 1.0e-03

        # Mesh
        self.print_debug("Loading mesh...")
        self.resource_dir = os.path.join(os.path.dirname(__file__), 'resources')
        self.meshfile = os.path.join(self.resource_dir, 'meshes', 'Tohoku{:d}.msh'.format(level))
        self.default_mesh = mesh or Mesh(self.meshfile)
        self.print_debug("Done!")

        # Fields
        self.set_initial_surface()
        self.set_bathymetry()

        # Timestepping: export once per minute for 24 minutes
        self.timestepper = 'CrankNicolson'
        # self.timestepper = 'SSPRK33'
        self.dt = 5.0
        # self.dt = 0.01
        self.dt_per_export = int(60.0/self.dt)
        self.start_time = 15*60.0
        self.end_time = 24*60.0
        # self.end_time = 60*60.0

        # Gauges where we have timeseries
        self.gauges = {
            "P02": {"lonlat": (142.5016, 38.5002)},
            "P06": {"lonlat": (142.5838, 38.6340)},
            "801": {"lonlat": (141.6856, 38.2325)},
            "802": {"lonlat": (142.0969, 39.2586)},
            "803": {"lonlat": (141.8944, 38.8578)},
            "804": {"lonlat": (142.1867, 39.6272)},
            "806": {"lonlat": (141.1856, 36.9714)},
        }
        # TODO: Use converted pressure timeseries
        self.gauges["P02"]["data"] = [0.00, 0.07, 0.12, 0.46, 0.85, 1.20, 1.55, 1.90, 2.25, 2.50,
                                      2.80, 3.10, 3.90, 4.80, 4.46, 2.25, -0.45, -0.17, -1.60,
                                      -0.82, -0.44, -0.26, -0.08, 0.13, 0.42, 0.71]
        self.gauges["P06"]["data"] = [0.00, 0.10, 0.30, 0.65, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55,
                                      2.90, 3.50, 4.50, 4.85, 3.90, 1.55, -0.35, -1.05, -0.65,
                                      -0.30, -0.15, 0.05, 0.18, 0.35, 0.53, 0.74]

        # Possible coastal locations of interest, including major cities and nuclear power plants
        locations_of_interest = {
            "Fukushima Daiichi": {"lonlat": (141.0281, 37.4213)},
            "Onagawa": {"lonlat": (141.5008, 38.3995)},
            "Fukushima Daini": {"lonlat": (141.0249, 37.3166)},
            "Tokai": {"lonlat": (140.6067, 36.4664)},
            "Hamaoka": {"lonlat": (138.1433, 34.6229)},
            "Tohoku": {"lonlat": (141.3903, 41.1800)},
            "Tokyo": {"lonlat": (139.6917, 35.6895)},
        }
        self.locations_of_interest = {loc: locations_of_interest[loc] for loc in locations}
        radii = {locations[i]: r for i, r in enumerate(radii)}

        # Convert coordinates to UTM and create timeseries array
        for loc in (self.gauges, self.locations_of_interest):
            for l in loc:
                loc[l]["timeseries"] = []
                lon, lat = loc[l]["lonlat"]
                loc[l]["utm"] = from_latlon(lat, lon, force_zone_number=54)
                loc[l]["coords"] = loc[l]["utm"]

        # Regions of interest
        loi = self.locations_of_interest
        self.region_of_interest = [loi[loc]["coords"] + (radii[loc], ) for loc in loi]

    def read_bathymetry_file(self):
        self.print_debug("Reading bathymetry file...")
        nc = netCDF4.Dataset(os.path.join(self.resource_dir, 'bathymetry', 'bathymetry.nc'), 'r')
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:-1]
        elev = nc.variables['elevation'][:-1, :]
        nc.close()
        self.print_debug("Done!")
        return lon, lat, elev

    def read_surface_file(self, zeroed=True):
        self.print_debug("Reading initial surface file...")
        fname = 'surf'
        if zeroed:
            fname = '_'.join([fname, 'zeroed'])
        nc = netCDF4.Dataset(os.path.join(self.resource_dir, 'surf', fname + '.nc'), 'r')
        lon = nc.variables['lon' if zeroed else 'x'][:]
        lat = nc.variables['lat' if zeroed else 'y'][:]
        elev = nc.variables['z'][:, :]
        nc.close()
        self.print_debug("Done!")
        return lon, lat, elev

    def set_boundary_conditions(self, fs):
        ocean_tag = 100
        coast_tag = 200
        fukushima_tag = 300
        self.boundary_conditions = {
            coast_tag: {'un': Constant(0.0)},
            fukushima_tag: {'un': Constant(0.0)},
            ocean_tag: {'un': Constant(0.0), 'elev': Constant(0.0)},  # Weakly reflective boundaries
        }
        # TODO: Sponge at ocean boundary?
        #        - Could potentially do this by defining a gradation to the ocean boundary with a
        #          different PhysID.
        return self.boundary_conditions

    def get_update_forcings(self, solver_obj):
        def update_forcings(t):
            # self.print_debug("#### DEBUG t: {:.2f}".format(t))
            return
        return update_forcings

    def annotate_plot(self, axes, coords="lonlat", gauges=False):
        """
        Annotate a plot on axes `axes` in coordinate system `coords` with all gauges or locations of
        interest, as determined by the Boolean kwarg `gauges`.
        """
        coords = coords or "lonlat"
        try:
            assert coords in ("lonlat", "utm")
        except AssertionError:
            raise ValueError("Coordinate system {:s} not recognised.".format(coords))
        dat = self.gauges if gauges else self.locations_of_interest
        for loc in dat:
            x, y = dat[loc][coords]
            xytext = (x + 0.3, y)
            color = "indigo"
            if loc == "P02":
                color = "navy"
                xytext = (x + 0.5, y - 0.4)
            elif loc == "P06":
                color = "navy"
                xytext = (x + 0.5, y + 0.2)
            elif "80" in loc:
                color = "darkgreen"
                xytext = (x - 0.8, y)
            elif loc == "Fukushima Daini":
                continue
            elif loc == "Fukushima Daiichi":
                loc = "Fukushima"
            elif loc in ("Tokyo", "Hamaoka"):
                xytext = (x + 0.3, y-0.6)
            ha = "center" if gauges else "left"
            axes.annotate(loc, xy=(x, y), xytext=xytext, fontsize=10, color=color, ha=ha)
            circle = plt.Circle((x, y), 0.1, color=color)
            axes.add_patch(circle)
