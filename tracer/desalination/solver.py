from thetis import *

from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveDesalinationProblem"]


class AdaptiveDesalinationProblem(AdaptiveProblem):
    """General solver object for adaptive desalination outfall problems."""
    # TODO: doc
    def __init__(self, *args, **kwargs):
        super(AdaptiveDesalinationProblem, self).__init__(*args, **kwargs)
        self.ramp_dir = kwargs.get('ramp_dir')
        self.load_mesh = kwargs.get('load_mesh')

    def set_initial_condition(self):
        if self.op.spun:
            self.load_state(0, self.ramp_dir)
            if self.load_mesh is not None:
                tmp = self.fwd_solutions[0].copy(deepcopy=True)
                u_tmp, eta_tmp = tmp.split()
                self.set_meshes(self.load_mesh)
                self.setup_all()
                u, eta = self.fwd_solutions[0].split()
                u.project(u_tmp)
                eta.project(eta_tmp)
        else:
            super(AdaptiveDesalinationProblem, self).set_initial_condition()
