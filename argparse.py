import argparse


__all__ = ["ArgumentParser"]


# TODO: Integrate into Options parameter classes - these define the defaults
class ArgumentParser(argparse.ArgumentParser):
    # TODO: Doc
    def __init__(self, *args, **kwargs):
        super(ArgumentParser, self).__init__(*args, **kwargs)

        # Args
        if kwargs.get('basis', False):
            self.add_basis_args()

        # Kwargs
        if kwargs.get('plotting', False):
            self.add_plotting_args()
        if kwargs.get('shallow_water', False):
            self.add_shallow_water_args()

        # Flags for any Firedrake script
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
