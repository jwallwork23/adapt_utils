"""
Runs the unsteady test case scripts found in `unsteady/test_cases/`. In some cases these examples include
assertions which verify desired behaviour, but in most cases it is just verified that the solve does not
crash and reaches the end of the required time period.

The code used in this script was largely copied from `thetis/test/examples.py`.
"""
import pytest
import os
import sys
import glob
import subprocess
import shutil


examples = [
    'cosine_prescribed_velocity/run.py',
    'solid_body_rotation/run_fixed_mesh.py',
    'solid_body_rotation/run_lagrangian.py',
    # 'bubble_shear/run.py',  # TODO: Currently requires an 'interpretation' input parameter
    'rossby_wave/run_fixed_mesh.py',
    # 'rossby_wave/run_moving_mesh.py',  # TODO: Currently takes too long to run
    'balzano/run_fixed_mesh.py',
    'balzano/run_moving_mesh.py',
    # 'trench/run_fixed_mesh.py',  # TODO
    # 'trench/run_moving_mesh.py',  # TODO
    # 'turbine_array/run.py',  # TODO
]

cwd = os.path.abspath(os.path.dirname(__file__))
unsteady_dir = os.path.abspath(os.path.join(cwd, '..', 'unsteady', 'test_cases'))
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
