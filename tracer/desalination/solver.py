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

            # Load hydrodynamics from file
            self.op.solve_tracer = False
            self.load_state(0, self.ramp_dir)
            if self.load_mesh is not None:
                tmp = self.fwd_solutions[0].copy(deepcopy=True)
                u_tmp, eta_tmp = tmp.split()
                self.set_meshes(self.load_mesh)
                self.setup_all()
                u, eta = self.fwd_solutions[0].split()
                u.project(u_tmp)
                eta.project(eta_tmp)

            # Set background salinity
            self.op.solve_tracer = True
            self.fwd_solution_tracer.assign(self.op.background_salinity)
        else:
            super(AdaptiveDesalinationProblem, self).set_initial_condition()
