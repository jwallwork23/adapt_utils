import numpy as np
import os

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.misc import box
from adapt_utils.swe.tsunami.conversion import from_latlon


__all__ = ["TohokuHazardOptions"]


class TohokuHazardOptions(TohokuOkadaBasisOptions):
    """
    Parameter class for hazard assessment applications for the Tohoku tsunami case study,
    starting with the initial condition deduced in the inversion section.

    The hazard being assessed is determined by the QoI. By default, we integrate over only a
    subinterval of the time period, starting with :attr:`start_time`.
    """
    def __init__(self, *args, kernel_shape='gaussian', **kwargs):
        """
        :kwarg kernel_shape: shape to use for region of interest, from
            {'gaussian', 'circular_bump', 'ball'}.
        """
        super(TohokuHazardOptions, self).__init__(*args, **kwargs)

        # Choose kernel function type
        supported_kernels = ('gaussian', 'circular_bump', 'ball')
        if kernel_shape not in supported_kernels:
            raise ValueError("Please choose kernel_shape from {:}.".format(supported_kernels))
        self.kernel_shape = kernel_shape
        self.kernel_function = self.__getattribute__(kernel_shape)
        self.start_time = kwargs.get('start_time', 1200.0)
        self.end_time = kwargs.get('end_time', 1440.0)
        assert self.start_time < self.end_time

        # Location classifications
        self.locations_to_consider = kwargs.get('locations', ['Fukushima Daiichi'])
        self.get_locations_of_interest(**kwargs)

        # Load results of inversion
        fpath = os.path.join(os.path.dirname(__file__), '..', 'inversion', 'okada', 'data')
        level = kwargs.get('level', 0)
        inversion_level = kwargs.get('inversion_level', level)
        fname = os.path.join(fpath, 'opt_progress_discrete_{:d}_all_ctrl.npy'.format(inversion_level))
        opt_controls = np.load(fname)[-1]
        self.control_parameters['slip'] = opt_controls[0::2]
        self.control_parameters['rake'] = opt_controls[1::2]

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
            "Ogasawara": {"lonlat": (142.1918, 27.0944)},
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
        prob.kernels[i] = Function(prob.V[i], name="QoI kernel")
        kernel_u, kernel_eta = prob.kernels[i].split()
        kernel_u.rename("QoI kernel (velocity component)")
        kernel_eta.rename("QoI kernel (elevation component)")
        kernel_eta.interpolate(self.kernel_function(prob.meshes[i], source=False))

    def _get_update_forcings_forward(self, prob, i):
        return lambda t: None

    def _get_update_forcings_adjoint(self, prob, i):
        return lambda t: None

    def get_regularisation_term(self, prob):
        raise NotImplementedError

    def set_initial_condition(self, prob, **kwargs):
        self.default_mesh = prob.mesh
        self.get_subfaults(reset=True)
        super(TohokuHazardOptions, self).set_initial_condition(prob, **kwargs)

    def annotate_plot(self, axes, coords="utm", fontsize=12, textcolour='r', markercolour='r'):
        """
        Annotate `axes` in coordinate system `coords` with all locations of interest.

        :arg axes: `matplotlib.pyplot` :class:`axes` object.
        :kwarg coords: coordinate system, from 'lonlat' and 'utm'.
        :kwarg fontsize: font size to use in annotations.
        """
        if coords not in ("lonlat", "utm"):
            raise ValueError("Coordinate system {:s} not recognised.".format(coords))
        for loc in self.locations_of_interest:
            offset = 1000.0e+03 if 'Fukushima' in loc else 650.0e+03
            x, y = np.copy(self.locations_of_interest[loc][coords])
            kwargs = {
                "xy": self.locations_of_interest[loc][coords],
                "color": textcolour,
                "ha": "right",
                "va": "center",
                "fontsize": fontsize,
            }
            x += offset
            kwargs["xytext"] = (x, y)
            axes.plot(*self.locations_of_interest[loc][coords], '*', color=markercolour)
            axes.annotate(loc, **kwargs)
