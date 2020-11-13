import numpy as np

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.swe.tsunami.conversion import from_latlon


__all__ = ["TohokuHazardOptions"]


class TohokuHazardOptions(TohokuOptions):
    # TODO: doc
    def __init__(self, *args, kernel_shape='gaussian', **kwargs):
        """
        :kwarg radius: distance indicating radii around the locations of interest, thereby
            determining regions of interest for use in hazard assessment QoIs.
        """
        super(TohokuHazardOptions, self).__init__(*args, **kwargs)

        # Choose kernel function type
        supported_kernels = ('gaussian', 'circular_bump', 'ball')
        if kernel_shape not in supported_kernels:
            raise ValueError("Please choose kernel_shape from {:}.".format(supported_kernels))
        self.kernel_shape = kernel_shape
        self.kernel_function = self.__getattribute__(kernel_shape)

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
        radius = kwargs.get('radius', 100.0e+03)
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

    def set_qoi_kernel(self, prob, i):
        from firedrake import assemble, Constant, dx, Function

        # Evaluate kernel function
        b = self.kernel_function(prob.meshes[i], source=False)

        # Normalise by area computed on fine reference mesh (if available)
        locations = list(self.locations_of_interest.keys())
        r = self.radius
        if locations == ['Fukushima Daiichi', ] and np.isclose(r, 50e+03):
            area_fine_mesh = {
                'ball': 4.07920324e+09,
                'circular_bump': 3.30611745e+09,
                'gaussian': 4.18761028e+09,
            }[self.kernel_shape]
            area = assemble(b*dx)
            rescaling = Constant(1.0 if np.allclose(area, 0.0) else area_fine_mesh/area)
        elif locations == ['Fukushima Daiichi', ] and np.isclose(r, 100e+03):
            area_fine_mesh = {
                'ball': 1.73829649e+10,
                'circular_bump': 1.38578518e+10,
                'gaussian': 1.66879811e+10,
            }[self.kernel_shape]
            area = assemble(b*dx)
            rescaling = Constant(1.0 if np.allclose(area, 0.0) else area_fine_mesh/area)
        else:
            rescaling = Constant(1.0)

        prob.kernels[i] = Function(prob.V[i], name="QoI kernel")
        kernel_u, kernel_eta = prob.kernels[i].split()
        kernel_u.rename("QoI kernel (velocity component)")
        kernel_eta.rename("QoI kernel (elevation component)")
        kernel_eta.interpolate(rescaling*b)

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
        offset = 40.0e+03
        for loc in self.locations_of_interest:
            x, y = np.copy(self.locations_of_interest[loc][coords])
            kwargs = {
                "xy": self.locations_of_interest[loc][coords],
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
            axes.plot(*self.locations_of_interest[loc][coords], 'x', color=kwargs["color"])
            axes.annotate(loc, **kwargs)
