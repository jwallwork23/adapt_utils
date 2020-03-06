from firedrake import *

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

from adapt_utils.adapt.kernels import *
from adapt_utils.options import Options


__all__ = ["AdaptiveMesh"]


class AdaptiveMesh():
    """
    Wrapper which adds extra features to mesh.
    """
    def __init__(self, mesh, levels=0, op=Options()):
        """
        `AdaptiveMesh` object is initialised as the basis of a `MeshHierarchy`.
        """
        self.levels = levels
        self.op = op
        self.mesh = mesh
        if mesh.__class__.__name__ == 'HierarchyBase':
            self.hierarchy = mesh
        elif levels > 0:
            self.hierarchy = MeshHierarchy(mesh, levels)
        if hasattr(self, 'hierarchy'):
            n = len(self.hierarchy)
            self.mesh = self.hierarchy[n-levels-1]
            if levels > 0:
                self.refined_mesh = self.hierarchy[n-levels]
        self.dim = self.mesh.topological_dimension()
        assert self.dim in (2, 3)

        self.n = FacetNormal(self.mesh)
        if self.dim == 2:
            self.tangent = as_vector([-self.n[1], self.n[0]])  # Tangent vector
        elif self.dim == 3:
            warnings.warn("#### TODO: 3D mesh tangent vector not implemented")
            # raise NotImplementedError  # TODO: Get a tangent vector in 3D
        else:
            raise NotImplementedError
        self.facet_area = FacetArea(self.mesh)
        self.h = CellSize(self.mesh)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P0_vec = VectorFunctionSpace(self.mesh, "DG", 0)
        self.P0_ten = TensorFunctionSpace(self.mesh, "DG", 0)
        self.jacobian_sign = interpolate(sign(JacobianDeterminant(self.mesh)), self.P0)

    def copy(self):  # FIXME: Doesn't preserve hierarchy
        return AdaptiveMesh(Mesh(Function(self.mesh.coordinates)), levels=self.levels)

    def get_quality(self):  # FIXME: Why do rotations of the same element not have the same quality?
        """
        Compute the scaled Jacobian for each mesh element:
    ..  math::
            Q(K) = \frac{\det(J_K)}{\|\mathbf e_1\|\,\|\mathbf e2\|},

        where element :math:`K` is defined by edges :math:`\mathbf e_1` and :math:`\mathbf e_2`.

        NOTE that :math:`J_K = [\mathbf e_1, \mathbf e_2]`.
        """
        assert self.dim == 2
        J = interpolate(Jacobian(self.mesh), self.P0_ten)
        unswapped = as_matrix([[J[0, 0], J[0, 1]], [J[1, 0], J[1, 1]]])
        swapped = as_matrix([[J[1, 0], J[1, 1]], [J[0, 0], J[0, 1]]])
        sgn = Function(self.P0)
        sgn.dat.data[:] = self.jacobian_sign.dat.data
        # J.interpolate(conditional(ge(self.jacobian_sign, 0), unswapped, swapped))
        J.interpolate(conditional(ge(sgn, 0), unswapped, swapped))
        detJ = det(J)
        edge1 = as_vector([J[0, 0], J[1, 0]])
        edge2 = as_vector([J[0, 1], J[1, 1]])
        norm1 = sqrt(dot(edge1, edge1))
        norm2 = sqrt(dot(edge2, edge2))

        self.scaled_jacobian = interpolate(detJ/(norm1*norm2), self.P0)
        return self.scaled_jacobian

    def check_inverted(self, error=True):
        r = interpolate(JacobianDeterminant(self.mesh)/self.jacobian_sign, self.P0)
        if r.vector().gather().min() < 0:
            if error:
                raise ValueError("ERROR! Mesh has inverted elements!")
            else:
                warnings.warn("WARNING! Mesh has inverted elements!")

    def plot_quality(self, ax=None, savefig=True):
        """
        Plot scaled Jacobian using a discretised scale:
          * green   : high quality elements (over 75%);
          * yellow  : medium quality elements (50 - 75%);
          * blue    : low quality elements (0 - 50%);
          * magenta : inverted elements (quality < 0).
        """
        self.get_quality()

        # FIXME: Inverted elements do not show! Tried making transparent but it didn't do anything.
        cmap = plt.get_cmap('viridis', 30)
        newcolours = cmap(np.linspace(0, 1, 30))
        newcolours[:10] = np.array([1, 0, 1, 1])    # Magenta
        newcolours[10:20] = np.array([0, 1, 1, 1])  # Cyan
        newcolours[20:25] = np.array([1, 1, 0, 1])  # Yellow
        newcolours[25:] = np.array([0, 1, 0, 1])    # Green
        newcmap = ListedColormap(newcolours)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        cax = plot(self.scaled_jacobian, colorbar=True, vmin=-0.5, vmax=1, cmap=newcmap, axes=ax)
        ax.set_title("Scaled Jacobian")
        plot(self.mesh, axes=ax)
        if savefig:
            plt.savefig(os.path.join(self.op.di, 'scaled_jacobian.pdf'))

    def save_plex(self, filename):
        """
        Save mesh in DMPlex format.
        """
        viewer = PETSc.Viewer().createHDF5(filename, 'r')
        viewer(self.mesh._plex)

    def load_plex(self, filename):
        """
        Load mesh from DMPlex format. The `MeshHierarchy` is reinstated.
        """
        newplex = PETSc.DMPlex().create()
        newplex.createFromFile(filename)
        self.__init__(Mesh(newplex), levels=self.levels)

    def pragmatic_adapt(self, metric):
        """
        Adapt mesh using a specified metric. The `MeshHierarchy` is reinstated.
        """
        self.__init__(adapt(self.mesh, metric), levels=self.levels)

    def get_edge_lengths(self):
        """
        For each element, find the lengths of associated edges, stored in a HDiv trace field.

        NOTE: The plus sign is arbitrary and could equally well be chosen as minus.
        """
        HDivTrace = FunctionSpace(self.mesh, "HDiv Trace", 0)
        v, u = TestFunction(HDivTrace), TrialFunction(HDivTrace)
        self.edge_lengths = Function(HDivTrace, name="Edge lengths")
        mass_term = v('+')*u('+')*dS + v*u*ds
        rhs = v('+')*self.facet_area*dS + v*self.facet_area*ds
        solve(mass_term == rhs, self.edge_lengths)

    def get_edge_vectors(self):
        """
        For each element, find associated edge vectors, stored in a HDiv trace field.

        NOTES:
          * The plus sign is arbitrary and could equally well be chosen as minus.
          * The sign of the returned vectors is arbitrary and could equally well take the minus sign.
        """
        HDivTrace_vec = VectorFunctionSpace(self.mesh, "HDiv Trace", 0)
        v, u = TestFunction(HDivTrace_vec), TrialFunction(HDivTrace_vec)
        self.edge_vectors = Function(HDivTrace_vec, name="Edge vectors")
        mass_term = inner(v('+'), u('+'))*dS + inner(v, u)*ds
        rhs = inner(v('+'), self.tangent('+')*self.facet_area)*dS
        rhs += inner(v, self.tangent*self.facet_area)*ds
        solve(mass_term == rhs, self.edge_vectors)

    def get_maximum_length_edge(self):
        """
        For each element, find the associated edge of maximum length.
        """
        self.get_edge_lengths()
        self.get_edge_vectors()
        self.maximum_length_edge = Function(self.P0_vec, name="Maximum length edge")
        par_loop(get_maximum_length_edge(self.dim), dx, {'edges': (self.edge_lengths, READ),
                                                         'vectors': (self.edge_vectors, READ),
                                                         'max_vector': (self.maximum_length_edge, RW)
                                                        })

    def get_cell_metric(self):  # FIXME: Something doesn't seem right
        """
        Compute cell metric associated with mesh.

        Based on code by Lawrence Mitchell.
        """
        J = interpolate(Jacobian(self.mesh), self.P0_ten)
        self.cell_metric = Function(self.P0_ten, name="Cell metric")
        kernel = eigen_kernel(singular_value_decomposition, self.dim)
        op2.par_loop(kernel, self.P0_ten.node_set, self.cell_metric.dat(op2.INC), J.dat(op2.READ))

    def get_cell_size(self, u, mode='nguyen'):
        """
        Measure of element size recommended in [Nguyen et al., 2009]: maximum edge length, projected
        onto the velocity field `u`.
        """
        try:
            assert isinstance(u, Constant) or isinstance(u, Function)
        except AssertionError:
            raise ValueError("Velocity field should be either `Function` or `Constant`.")
        if mode == 'diameter':
            self.cell_size = interpolate(self.h, self.P0)
        elif mode == 'nguyen':
            self.get_maximum_length_edge()
            v = self.maximum_length_edge
            self.cell_size = interpolate(abs((u[0]*v[0] + u[1]*v[1]))/sqrt(dot(u, u)), self.P0)
        elif mode == 'cell_metric':
            self.get_cell_metric()
            self.cell_size = interpolate(dot(u, dot(self.cell_metric, u)), self.P0)
        else:
            raise ValueError("Element measure must be chosen from {'diameter', 'nguyen', 'cell_metric'}.")

    def subdomain_indicator(self, subdomain_id):
        """Creates a P0 indicator function relating with `subdomain_id`."""
        return assemble(TestFunction(self.P0)*dx(subdomain_id))
