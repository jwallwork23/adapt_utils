import numpy as np

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.unsteady.swe.tsunami.conversion import from_latlon


__all__ = ["TohokuHazardOptions"]


class TohokuHazardOptions(TohokuOptions):
    # TODO: doc
    def __init__(self, *args, **kwargs):
        """
        :kwarg radius: distance indicating radii around the locations of interest, thereby
            determining regions of interest for use in hazard assessment QoIs.
        """
        super(TohokuHazardOptions, self).__init__(*args, **kwargs)

        # Timestepping
        # ============
        #   * The parent class is geared up for gauge timeseries inversion and therefore uses a
        #     2 hour simulation period.
        #   * In this class we are only interested in the tsunami's approach of the coast and
        #     therefore use a reduced time window.
        self.start_time = kwargs.get('start_time', 0.0)
        self.end_time = kwargs.get('end_time', 24*60.0)
        # self.end_time = kwargs.get('end_time', 60*60.0)

        # Location classifications
        self.locations_to_consider = (
            # "Onagawa",
            # "Tokai",
            # "Hamaoka",
            # "Tohoku",
            # "Tokyo",
            "Fukushima Daiichi",
            # "Fukushima Daini",
        )
        self.get_locations_of_interest(**kwargs)

    def get_locations_of_interest(self, **kwargs):
        """
        Read in locations of interest, determine their coordinates and check these coordinate lie
        within the domain.

        The possible coastal locations of interest include major cities and nuclear power plants:

        * Cities:
          - Onagawa;
          - Tokai;
          - Hamaoka;
          - Tohoku;
          - Tokyo.

        * Nuclear power plants:
          - Fukushima Daiichi;
          - Fukushima Daini.
        """
        radius = kwargs.get('radius', 50.0e+03)
        locations_of_interest = {
            "Onagawa": {"lonlat": (141.5008, 38.3995)},
            "Tokai": {"lonlat": (140.6067, 36.4664)},
            "Hamaoka": {"lonlat": (138.1433, 34.6229)},
            "Tohoku": {"lonlat": (141.3903, 41.1800)},
            "Tokyo": {"lonlat": (139.6917, 35.6895)},
            "Fukushima Daiichi": {"lonlat": (141.0281, 37.4213)},
            "Fukushima Daini": {"lonlat": (141.0249, 37.3166)},
        }

        # Convert coordinates to UTM and create timeseries array
        self.locations_of_interest = {}
        for loc in self.locations_to_consider:
            self.locations_of_interest[loc] = {"data": [], "timeseries": []}
            self.locations_of_interest[loc]["lonlat"] = locations_of_interest[loc]["lonlat"]
            lon, lat = self.locations_of_interest[loc]["lonlat"]
            self.locations_of_interest[loc]["utm"] = from_latlon(lat, lon, force_zone_number=54)
            self.locations_of_interest[loc]["coords"] = self.locations_of_interest[loc]["utm"]

        # Regions of interest
        loi = self.locations_of_interest
        self.region_of_interest = [loi[loc]["coords"] + (radius, ) for loc in loi]

    def _get_update_forcings_forward(self, prob, i):

        def update_forcings(t):
            return

        return update_forcings

    def _get_update_forcings_adjoint(self, prob, i):

        def update_forcings(t):
            return

        return update_forcings

    def get_regularisation_term(self, prob):
        raise NotImplementedError

    def annotate_plot(self, axes, coords="utm", fontsize=12):
        """
        Annotate `axes` in coordinate system `coords` with all locations of interest.

        :arg axes: `matplotlib.pyplot` :class:`axes` object.
        :kwarg coords: coordinate system, from 'lonlat' and 'utm'.
        :kwarg fontsize: font size to use in annotations.
        """
        if coords not in ("lonlat", "utm"):
            raise ValueError("Coordinate system {:s} not recognised.".format(coords))
        dat = self.gauges if gauges else self.locations_of_interest
        offset = 40.0e+03
        for loc in dat:
            x, y = np.copy(dat[loc][coords])
            kwargs = {
                "xy": dat[loc][coords],
                "color": "indigo",
                "ha": "right",
                "va": "center",
                "fontsize": fontsize,
            }
            if loc == "Fukushima Daini":
                continue
            elif loc == "Fukushima Daiichi":
                loc = "Fukushima"
            x += offset
            kwargs["xytext"] = (x, y)
            axes.plot(*dat[loc][coords], 'x', color=kwargs["color"])
            axes.annotate(loc, **kwargs)