# from adapt_utils.adapt.adaptation import AdaptiveMesh
from adapt_utils.solver import UnsteadyProblem


__all__ = ["AdaptiveUnsteadyProblem"]


class AdaptiveUnsteadyProblem(UnsteadyProblem):
    # TODO: doc
    def __init__(self, *args, **kwargs):
        super(AdaptiveUnsteadyProblem, self).__init__(*args, **kwargs)
        raise NotImplementedError  # TODO

    def solve_step(self, adjoint=False):
        """
        Solve forward PDE on a particular mesh.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.
        """
        raise NotImplementedError

    def quantity_of_interest(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        raise NotImplementedError("Should be implemented in derived class.")
