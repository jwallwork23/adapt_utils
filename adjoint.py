import firedrake
import ufl
from pyadjoint.tape import no_annotations, get_working_tape, stop_annotating
from pyadjoint.enlisting import Enlist


__all__ = ["compute_adj_sol"]


backend = firedrake


# TODO: Surely we don't need a control
def compute_adj_sol(J, m, tape=None, adj_value=1.0, last_block=0, enriched_space=None, kernel=None):
    """
    Compute the adjoint solution w.r.t. J with respect to the initialisation value of m,
    that is the value of m at its creation. The adjoint solution values are stored in
    the `SolveBlock`s of the tape.

    Args:
        J (AdjFloat):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        tape: The tape to use. Default is the current tape.
    """
    tape = get_working_tape() if tape is None else tape
    tape.reset_variables()
    J.adj_value = adj_value
    m = Enlist(m)

    used_kernel = False
    if enriched_space is not None:
        assert kernel is not None
    else:
        used_kernel = True

    sbs = ('SolveBlock', 'LinearVariationalSolveBlock', 'NonlinearVariationalSolveBlock')

    with stop_annotating():
        with tape.marked_nodes(m):
            for i in range(len(tape._blocks)-1, last_block - 1, -1):
                if tape._blocks[i].__class__.__name__ in sbs and enriched_space is not None:
                    if not used_kernel:  # Assign adjoint 'initial condition'
                        tape._blocks[i].kernel = kernel
                        used_kernel = True
                    tape._blocks[i].enriched_space = enriched_space
                    tape._blocks[i] = evaluate_adj(tape._blocks[i], markings=True)
                else:
                    tape._blocks[i].evaluate_adj(markings=True)

@no_annotations
def evaluate_adj(solve_block, markings=False):
    outputs = solve_block.get_outputs()
    adj_inputs = []
    has_input = False
    for output in outputs:
        adj_inputs.append(output.adj_value)
        if output.adj_value is not None:
            has_input = True

    if not has_input:
        return

    deps = solve_block.get_dependencies()
    inputs = [bv.saved_output for bv in deps]
    relevant_dependencies = [(i, bv) for i, bv in enumerate(deps) if bv.marked_in_path or not markings]

    if len(relevant_dependencies) <= 0:
        return

    # ~~~~ prepare_evaluate_adj ~~~~
    fwd_block_variable = outputs[0]
    u = fwd_block_variable.output

    if hasattr(solve_block, 'kernel'):
        assert hasattr(solve_block, 'enriched_space')
        dJdu = backend.assemble(backend.inner(solve_block.kernel, backend.TestFunction(solve_block.enriched_space))*backend.dx()).vector()
    else:
        dJdu = adj_inputs[0]

    # Modify arguments of weak form to correspond to enriched space
    if hasattr(solve_block, 'enriched_space'):
        solve_block.function_space = solve_block.enriched_space  # FIXME: this breaks forward solver
        test = backend.TestFunction(solve_block.enriched_space)
        trial = backend.TrialFunction(solve_block.enriched_space)
        if solve_block.linear:
            solve_block.rhs = ufl.replace(solve_block.rhs, {solve_block.rhs.arguments()[0]: test})
            solve_block.lhs = ufl.replace(solve_block.lhs, {solve_block.lhs.arguments()[0]: test,
                                                            solve_block.lhs.arguments()[1]: trial})
        else:
            solve_block.lhs = ufl.replace(solve_block.lhs, {solve_block.lhs.arguments()[0]: test})
        # FIXME: Account for the case where we use a different mesh
    else:
        trial = backend.TrialFunction(solve_block.function_space)

    F_form = solve_block._create_F_form()

    # print(fwd_block_variable.saved_output)
    dFdu = backend.derivative(F_form,
                              fwd_block_variable.saved_output,
                              trial)
    dFdu_form = backend.adjoint(dFdu)
    dJdu = dJdu.copy()
    solve_block.adj_sol = solve_block._assemble_and_solve_adj_eq(dFdu_form, dJdu)[0]

    return solve_block
