from thetis import AttrDict

import argparse


__all__ = ["ArgumentParser"]


# TODO: Integrate into Options parameter classes - these define the defaults
class ArgumentParser(argparse.ArgumentParser):
    """
    Custom argument parsing class for use in `adapt_utils`.

    There are various pre-defined arguments, which may be selected as keyword arguments when
    instantiating the object.

    :kwarg adjoint: positional argument for 'discrete' vs 'continuous'.
    :kwarg basis: positional argument for 'box' vs 'radial' vs 'okada'.
    :kwarg basis: positional argument for a combination of the above.
    :kwarg optimisation: keyword arguments to do with gradient-based optimisation.
    :kwarg plotting: keyword arguments to do with plotting via `matplotlib`.
    :kwarg shallow_water: keyword arguments to do with solving the shallow water equations.
    """
    def __init__(self, *args, **kwargs):
        common_kwargs = (
            'adjoint',
            'bases',
            'basis',
            'optimisation',
            'plotting',
            'shallow_water',
        )
        self.kwargs = {kwarg: kwargs.pop(kwarg, False) for kwarg in common_kwargs}
        self.kwargs['equation'] = self.kwargs['shallow_water']  # TODO: or tracer or sediment, etc.
        super(ArgumentParser, self).__init__(*args, **kwargs)
        for arg in self.kwargs:
            if self.kwargs[arg]:
                self.__getattribute__('add_{:s}_args'.format(arg))()

        # I/O
        self.add_argument("-extension", help="""
            Extension for output directory. The directory name will have the form outputs_<ext>.
            """)

        # Debugging
        self.add_argument("-debug", help="Toggle debugging")
        self.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

    def parse_args(self):
        """
        Parse all parameters.
        """
        if not hasattr(self, '_args'):
            self._args = super(ArgumentParser, self).parse_args()
        return self._args

    @property
    def args(self):
        return self.parse_args()

    def add_adjoint_args(self):
        """
        Add a parameter to toggle between discrete and continuous adjoint.
        """
        self.add_argument("adjoint", help="""
            Choose adjoint approach from {'discrete', 'continuous'}.
            """)

    def add_bases_args(self):
        """
        Add a parameter for looping over the source bases used in tsunami source inversion.
        """
        self.add_argument("bases", help="""
            Basis types for inversion, chosen from {'box', 'radial', 'okada'} and separated by
            commas.
            """)

    def add_basis_args(self):
        """
        Add a parameter specifying the source basis used in tsunami source inversion.
        """
        self.add_argument("basis", help="Basis type for inversion, from {'box', 'radial', 'okada'}.")

    def basis_args(self):
        """
        Process parameter specifying the source basis used in tsunami source inversion.
        """
        basis = self.args.basis
        if basis not in ('box', 'radial', 'okada'):
            raise ValueError("Basis type '{:s}' not recognised.".format(basis))
        return basis

    def add_optimisation_args(self):
        """
        Add parameters to do with optimisation.
        """
        self.add_argument("-continuous_timeseries", help="""
            Toggle discrete or continuous timeseries data
            """)
        self.add_argument("-gtol", help="Gradient tolerance (default 1.0e-08)")
        self.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
        self.add_argument("-taylor_test", help="Toggle Taylor testing")

    def add_plotting_args(self):
        """
        Add parameters to do with plotting.
        """
        self.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
        self.add_argument("-plot_png", help="Toggle plotting to .png")
        self.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
        self.add_argument("-plot_only", help="Just plot using saved data")

    def plotting_args(self):
        """
        Process parameters to do with plotting.
        """
        plot_pdf = bool(self.args.plot_pdf or False)
        plot_png = bool(self.args.plot_png or False)
        plot_all = bool(self.args.plot_all or False)
        plot_only = bool(self.args.plot_only or False)
        if plot_only:
            plot_all = True
        if plot_all:
            plot_pdf = plot_png = True
        extensions = []
        if plot_pdf:
            extensions.append('pdf')
        if plot_png:
            extensions.append('png')
        plot_any = len(extensions) > 0
        return AttrDict({
            'all': plot_all,
            'any': plot_any,
            'pdf': plot_pdf,
            'png': plot_png,
            'only': plot_only,
            'extensions': extensions,
        })

    def add_shallow_water_args(self):
        """
        Add parameters to do with the shallow water solver.
        """
        self.add_argument("-family", help="Finite element pair")
        self.add_argument("-stabilisation", help="Stabilisation approach")
        self.add_argument("-nonlinear", help="Toggle nonlinear model")

    def add_equation_args(self):
        """
        Add parameters to do with solving any equation set.
        """
        self.add_argument("-dirty_cache", help="Dirty the cache to force compilations")
        self.add_argument("-end_time", help="End time of simulation")
        self.add_argument("-level", help="Mesh resolution level")
        self.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
