"""
Runs the unsteady test case scripts found in `unsteady/test_cases/`. In some cases these examples
include assertions which verify desired behaviour, but in most cases it is just verified that the
solve does not crash and reaches the end of the simulation.

The code used in this script was largely copied from `thetis/test/test_examples.py`.
"""
import pytest
import os
import sys
import glob
import subprocess
import shutil


# Set environment flag so indicate shorter tests
os.environ['REGRESSION_TEST'] = "1"

# Collate a list of all examples to be tested
examples = [

    # Fixed mesh
    'unsteady/test_cases/balzano/run_fixed_mesh.py',                # Has been cut short
    'unsteady/test_cases/beach_slope/run_fixed_mesh.py',            # Has been cut (very) short
    'unsteady/test_cases/beach_wall/run_fixed_mesh.py',             # Has been cut (very) short
    'unsteady/test_cases/idealised_desalination/run_fixed_mesh.py',
    'unsteady/test_cases/bubble_shear/run_fixed_mesh.py',           # Has been cut short
    'unsteady/test_cases/pulse_wave/run_fixed_mesh.py',             # Has been cut (very) short
    'unsteady/test_cases/rossby_wave/run_fixed_mesh.py',            # Has been cut short
    'unsteady/test_cases/solid_body_rotation/run_fixed_mesh.py',
    # 'unsteady/test_cases/spaceship/run_fixed_mesh.py',            # Takes too long to run
    'unsteady/test_cases/trench_1d/trench_hydro',
    'unsteady/test_cases/trench_1d/run_fixed_mesh.py',              # Has been cut (very) short
    'unsteady/test_cases/trench_slant/run_fixed_mesh.py',           # Has been cut (very) short
    # 'unsteady/test_cases/turbine_array/run_fixed_mesh.py',        # Takes too long to run
    'case_studies/tohoku/hazard/run_fixed_mesh.py',

    # Moving mesh
    'unsteady/test_cases/balzano/run_moving_mesh.py',               # Has been cut short
    # 'unsteady/test_cases/beach_slope/run_moving_mesh.py',         # TODO
    # 'unsteady/test_cases/beach_wall/run_moving_mesh.py',          # TODO
    # 'unsteady/test_cases/bubble_shear/run_lagrangian.py',         # TODO: xfail it
    # 'unsteady/test_cases/bubble_shear/run_moving_mesh.py',        # TODO
    # 'unsteady/test_cases/cosine_prescribed_velocity/compare.py',  # FIXME: Lagrangian coords don't match
    # 'unsteady/test_cases/pulse_wave/run_moving_mesh.py',          # TODO
    'unsteady/test_cases/rossby_wave/run_moving_mesh.py',           # Has been cut (very) short
    'unsteady/test_cases/solid_body_rotation/run_lagrangian.py',
    # 'unsteady/test_cases/trench_1d/run_moving_mesh.py',           # TODO
    # 'unsteady/test_cases/trench_slant/run_moving_mesh.py',        # TODO

    # Metric-based
    'case_studies/tohoku/hazard/run_adapt.py',
]
cwd = os.path.abspath(os.path.dirname(__file__))
unsteady_dir = os.path.abspath(os.path.join(cwd, '..'))
examples = [os.path.join(unsteady_dir, f) for f in examples]


@pytest.fixture(params=examples, ids=lambda x: os.path.basename(x))
def example_file(request):
    return os.path.abspath(request.param)


def test_examples(example_file, tmpdir, monkeypatch):
    assert os.path.isfile(example_file), "File '{:}' not found".format(example_file)

    # Copy any mesh files
    source = os.path.dirname(example_file)
    for f in glob.glob(os.path.join(source, '*.msh')):
        shutil.copy(f, str(tmpdir))

    # Change workdir to tmpdir
    monkeypatch.chdir(tmpdir)

    # Run example
    subprocess.check_call([sys.executable, example_file])
