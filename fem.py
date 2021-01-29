from firedrake import *


__all__ = ["cg2dg", "get_finite_element", "get_component_space", "get_component"]


def get_finite_element(fs, variant='equispaced'):
    """
    Extract :class:`FiniteElement` instance from a :class:`FunctionSpace`, with specified variant
    default.
    """
    el = fs.ufl_element()
    if hasattr(el, 'variant'):
        variant = el.variant() or variant
    return FiniteElement(el.family(), fs.mesh().ufl_cell(), el.degree(), variant=variant)


def get_component_space(fs, variant='equispaced'):
    """
    Extract a single (scalar) :class:`FunctionSpace` component from a :class:`VectorFunctionSpace`.
    """
    return FunctionSpace(fs.mesh(), get_finite_element(fs, variant=variant))


def get_component(f, index, component_space=None):
    """
    Extract component `index` of a :class:`Function` from a :class:`VectorFunctionSpace` and store
    it in a :class:`Function` defined on the appropriate (scalar) :class:`FunctionSpace`. The
    component space can either be provided or computed on-the-fly.
    """
    n = f.ufl_shape[0]
    try:
        assert index < n
    except AssertionError:
        raise IndexError("Requested index {:d} of a {:d}-vector.".format(index, n))

    # Create appropriate component space
    fi = Function(component_space or get_component_space(f.function_space()))

    # Transfer data
    par_loop(('{[i] : 0 <= i < v.dofs}', 's[i] = v[i, %d]' % index), dx,
             {'v': (f, READ), 's': (fi, WRITE)}, is_loopy_kernel=True)
    return fi


# --- Continuous to discontinuous transfer

def cg2dg(f_cg, f_dg=None):
    """
    Transfer data from a the degrees of freedom of a Pp field directly to those of the
    corresponding PpDG field, for some p>1.
    """
    n = len(f_cg.ufl_shape)
    assert f_cg.ufl_element().family() == 'Lagrange'
    if n == 0:
        _cg2dg_scalar(f_cg, f_dg)
    elif n == 1:
        _cg2dg_vector(f_cg, f_dg)
    elif n == 2:
        _cg2dg_tensor(f_cg, f_dg)
    else:
        raise NotImplementedError


def _cg2dg_scalar(f_cg, f_dg):
    fs = f_cg.function_space()
    f_dg = f_dg or Function(FunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree()))
    index = '{[i] : 0 <= i < cg.dofs}'
    kernel = 'dg[i] = cg[i]'
    par_loop((index, kernel), dx, {'cg': (f_cg, READ), 'dg': (f_dg, WRITE)}, is_loopy_kernel=True)


def _cg2dg_vector(f_cg, f_dg):
    fs = f_cg.function_space()
    f_dg = f_dg or Function(VectorFunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree()))
    index = '{[i, j] : 0 <= i < cg.dofs and 0 <= j < %d}' % f_cg.ufl_shape
    kernel = 'dg[i, j] = cg[i, j]'
    par_loop((index, kernel), dx, {'cg': (f_cg, READ), 'dg': (f_dg, WRITE)}, is_loopy_kernel=True)


def _cg2dg_tensor(f_cg, f_dg):
    fs = f_cg.function_space()
    f_dg = f_dg or Function(TensorFunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree()))
    index = '{[i, j, k] : 0 <= i < cg.dofs and 0 <= j < %d and 0 <= k < %d}' % f_cg.ufl_shape
    kernel = 'dg[i, j, k] = cg[i, j, k]'
    par_loop((index, kernel), dx, {'cg': (f_cg, READ), 'dg': (f_dg, WRITE)}, is_loopy_kernel=True)
