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
      * Bathymetry data extracted from GEBCO (https://www.gebco.net/).
      * Initial free surface elevation field generated by inversion on tide gauge data by
        [Saito et al.].
      * Timeseries for gauges P02 and P06 obtained via personal communication with T. Saito.
      * Timeseries for gauges 801-806 obtained from the Japanese Port and Airport Research
          Institute (PARI).
      * Timeseries for gauges KPG1 and KPG2 obtained from the Japanese Agency for Marine-Earth
          Science and Technology (JAMSTEC) via http://www.jamstec.go.jp/scdc/top_e.html.
      * Timeseries for gauge 21418 obtained from the US National Oceanic and Atmospheric
        Administration (NOAA) via https://www.ndbc.noaa.gov/station_page.php?station=21418.

    [Saito et al.] T. Saito, Y. Ito, D. Inazu, R. Hino, "Tsunami source of the 2011 Tohoku‐Oki
                   earthquake, Japan: Inversion analysis based on dispersive tsunami simulations",
                   Geophysical Research Letters (2011), 38(7).
    """
    def __init__(self, mesh=None, postproc=True, level=0, locations=["Fukushima Daiichi", ], radii=[50.0e+03, ], **kwargs):
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
        self.meshfile = os.path.join(self.resource_dir, 'meshes', 'Tohoku{:d}'.format(level))
        if mesh is None:
            if postproc:
                newplex = PETSc.DMPlex().create()
                newplex.createFromFile(self.meshfile + '.h5')
                self.default_mesh = Mesh(newplex)
            else:
                self.default_mesh = Mesh(self.meshfile + '.msh')
        else:
            self.default_mesh = mesh
        self.print_debug("Done!")

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
            "P02": {"lonlat": (142.5016, 38.5002), "operator": "Tohoku University"},  # TODO: depth
            "P06": {"lonlat": (142.5838, 38.6340), "operator": "Tohoku University"},  # TODO: depth
            "801": {"lonlat": (141.6856, 38.2325), "operator": "PARI"},  # TODO: depth
            "802": {"lonlat": (142.0969, 39.2586), "operator": "PARI"},  # TODO: depth
            "803": {"lonlat": (141.8944, 38.8578), "operator": "PARI"},  # TODO: depth
            "804": {"lonlat": (142.1867, 39.6272), "operator": "PARI"},  # TODO: depth
            "806": {"lonlat": (141.1856, 36.9714), "operator": "PARI"},  # TODO: depth
            "KPG1": {"lonlat": (144.4375, 41.7040), "depth": 2218.0, "operator": "JAMSTEC"},
            "KPG2": {"lonlat": (144.8485, 42.2365), "depth": 2210.0, "operator": "JAMSTEC"},
            "MPG1": {"lonlat": (134.4753, 32.3907), "depth": 2308.0, "operator": "JAMSTEC"},
            "MPG2": {"lonlat": (134.3712, 32.6431), "depth": 1507.0, "operator": "JAMSTEC"},
            # TODO: VCM1 (NEID)
            # TODO: VCM3 (NEID)
            "21418": {"lonlat": (148.655, 38.735), "depth": 5777.0, "operator": "NOAA"},
        }
        self.pressure_gauges = ("P02", "P06", "KPG1", "KPG2", "MPG1", "MPG2", "21418")
        # self.pressure_gauges += ("VCM1", "VCM3")
        self.gps_gauges = ("801", "802", "803", "804", "806")

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

    def set_boundary_conditions(self, prob, i):
        ocean_tag = 100
        coast_tag = 200
        fukushima_tag = 300
        boundary_conditions = {
            'shallow_water': {
                coast_tag: {'un': Constant(0.0)},
                fukushima_tag: {'un': Constant(0.0)},
                ocean_tag: {'un': Constant(0.0), 'elev': Constant(0.0)},  # Weakly reflective
            }
        }
        # TODO: Sponge at ocean boundary?
        #        - Could potentially do this by defining a gradation to the ocean boundary with a
        #          different PhysID.
        return boundary_conditions

    def annotate_plot(self, axes, coords="utm", gauges=False):
        """
        Annotate a plot on axes `axes` in coordinate system `coords` with all gauges or locations of
        interest, as determined by the Boolean kwarg `gauges`.
        """
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
            elif "PG" in loc or loc[0] == "2":
                color = "navy"
            elif loc == "Fukushima Daini":
                continue
            elif loc == "Fukushima Daiichi":
                loc = "Fukushima"
            elif loc in ("Tokyo", "Hamaoka"):
                xytext = (x + 0.3, y-0.6)
            ha = "center" if gauges else "left"
            axes.annotate(loc, xy=(x, y), xytext=xytext, fontsize=12, color=color, ha=ha)
            circle = plt.Circle((x, y), 0.1, color=color)
            axes.add_patch(circle)

    def get_gauge_data(self, gauge, sample=1):
        """Read gauge data for `gauge` from file, averaging over every `sample` points."""
        if gauge[0] == '8' and sample > 1:
            assert sample % 5 == 0
            sample //= 5
        time_prev = 0.0
        di = os.path.join(os.path.dirname(__file__), 'resources', 'gauges')
        fname = os.path.join(di, '{:s}.dat'.format(gauge))
        num_lines = sum(1 for line in open(fname, 'r'))
        self.gauges[gauge]['data'] = []
        self.gauges[gauge]['time'] = []
        with open(fname, 'r') as f:
            running = 0.0
            for i in range(num_lines):
                time, dat = f.readline().split()
                time = float(time)
                dat = float(dat)
                running += dat  # TODO: How to deal with NaNs?
                if sample == 1:
                    self.gauges[gauge]['time'].append(time)
                    self.gauges[gauge]['data'].append(dat)
                elif i % sample == 0 and i > 0:
                    if time < time_prev:
                        break  # FIXME
                    self.gauges[gauge]['time'].append(0.5*(time + time_prev))
                    self.gauges[gauge]['data'].append(running/sample)
                    running = 0.0
                    time_prev = time
