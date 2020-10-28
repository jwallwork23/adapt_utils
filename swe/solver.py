from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveProblem_hydrodynamics_only"]


class AdaptiveProblem_hydrodynamics_only(AdaptiveProblem):
    """
    General solver object for adaptive shallow water problems with no tracer, sediment or Exner
    component.
    """
    def __init__(self, *args, **kwargs):
        super(AdaptiveProblem_hydrodynamics_only, self).__init__(*args, **kwargs)
        if self.op.solve_tracer or self.op.solve_sediment or self.op.solve_exner:
            raise ValueError("This class is for problems with hydrodynamics only.")
