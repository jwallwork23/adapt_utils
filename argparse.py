import argparse


__all__ = ["ArgumentParser"]


# TODO: Integrate into Options parameter classes - these define the defaults
class ArgumentParser(argparse.ArgumentParser):
    # TODO: Doc
    def __init__(self, *args, **kwargs):
        self.kwargs = {
            'adjoint': kwargs.pop('adjoint', False),
            'basis': kwargs.pop('basis', False),
            'bases': kwargs.pop('bases', False),
            'plotting': kwargs.pop('plotting', False),
            'shallow_water': kwargs.pop('shallow_water', False),
        }
        super(ArgumentParser, self).__init__(*args, **kwargs)
        for arg in self.kwargs:
            if self.kwargs[arg]:
                self.__getattribute__('add_{:s}_args'.format(arg))()

        # I/O
        self.add_argument("-extension", help="""
            Extension for output directory. The directory name will have the form outputs_<ext>.
            """)
        self.add_argument("-plot_pvd", help="Toggle plotting to .pvd")

        # Debugging
        self.add_argument("-debug", help="Toggle debugging")
        self.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

    def parse_args(self):
        if not hasattr(self, '_args'):
            self._args = super(ArgumentParser, self).parse_args()
        return self._args

    @property
    def args(self):
        return self.parse_args()

    def add_adjoint_args(self):
        self.add_argument("adjoint", help="""
            Choose adjoint approach from {'discrete', 'continuous'}.
            """)

    def add_bases_args(self):
        self.add_argument("bases", help="""
            Basis types for inversion, chosen from {'box', 'radial', 'okada'} and separated by
            commas.
            """)

    def add_basis_args(self):
        self.add_argument("basis", help="Basis type for inversion, from {'box', 'radial', 'okada'}.")

    def basis_args(self):
        basis = self.args.basis
        if basis not in ('box', 'radial', 'okada'):
            raise ValueError("Basis type '{:s}' not recognised.".format(basis))
        return basis

    def add_plotting_args(self):
        self.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
        self.add_argument("-plot_png", help="Toggle plotting to .png")
        self.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
        self.add_argument("-plot_only", help="Just plot using saved data")

    def plotting_args(self):
        plot_pdf = bool(self.args.plot_pdf or False)
        plot_png = bool(self.args.plot_png or False)
        plot_all = bool(self.args.plot_all or False)
        plot_only = bool(self.args.plot_only or False)
        if plot_only:
            plot_all = True
        if plot_all:
            plot_pdf = plot_png = True
        return {
            'plot_pdf': plot_pdf,
            'plot_png': plot_png,
            'plot_all': plot_all,
            'plot_only': plot_only,
        }

    def add_shallow_water_args(self):
        self.add_argument("-family", help="Finite element pair")
        self.add_argument("-stabilisation", help="Stabilisation approach")
        self.add_argument("-nonlinear", help="Toggle nonlinear model")
