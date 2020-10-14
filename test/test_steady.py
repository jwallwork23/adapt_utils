"""
Runs the steady test case scripts found in `steady/test_cases/`. In some cases these examples include
assertions which verify desired behaviour, but in most cases it is just verified that the solve does not crash.

The code used in this script was largely copied from `thetis/test/test_examples.py`.
"""
import pytest
import os
import sys
import glob
import subprocess
import shutil


# Collate a list of all examples to be tested
examples = [
    # TODO: 'box_discharge2d/run_fixed_mesh.py',
    'point_discharge2d/run_fixed_mesh.py',
    # TODO: 'point_discharge2d/run_uniform_convergence.py',
    # TODO: 'point_discharge3d/run_fixed_mesh.py',
    # TODO: 'space_time_ripple/run_fixed_mesh.py',
    'turbine_array/run_fixed_mesh.py',
    # TODO: 'turbine_array/run_adapt.py',
    'turbine_array/run_uniform_convergence.py',
    # TODO: 'turbine_array/run_adaptive_convergence.py',
]
cwd = os.path.abspath(os.path.dirname(__file__))
unsteady_dir = os.path.abspath(os.path.join(cwd, '..', 'steady', 'test_cases'))
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
