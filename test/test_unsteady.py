"""
Runs the unsteady test case scripts found in `unsteady/test_cases/`. In some cases these examples
include assertions which verify desired behaviour, but in most cases it is just verified that the
solve does not crash and reaches the end of the simulation.

The code used in this script was largely copied from `thetis/test/examples.py`.
"""
import pytest
import os
import sys
import glob
import subprocess
import shutil


examples = [
    'balzano/run_fixed_mesh.py',
    'balzano/run_moving_mesh.py',  # TODO: Takes a while to run
    'beach_slope/run_fixed_mesh.py',
    # 'beach_slope/run_moving_mesh.py',  # TODO
    'beach_wall/run_fixed_mesh.py',
    # 'beach_wall/run_moving_mesh.py',  # TODO
    # 'bubble_shear/run.py',  # TODO: Currently requires an 'interpretation' input parameter
    'cosine_prescribed_velocity/run.py',  # FIXME
    'pulse_wave/run_fixed_mesh.py',
    # 'pulse_wave/run_moving_mesh.py',  # TODO
    'rossby_wave/run_fixed_mesh.py',  # TODO: Takes a while to run
    # 'rossby_wave/run_moving_mesh.py',  # TODO: Currently takes too long to run
    'solid_body_rotation/run_fixed_mesh.py',
    'solid_body_rotation/run_lagrangian.py',
    # 'spaceship/run_fixed_mesh.py',
    'trench_1d/run_fixed_mesh.py',
    # 'trench_1d/run_moving_mesh.py',  # TODO
    'trench_slant/run_fixed_mesh.py',
    # 'trench_slant/run_moving_mesh.py',  # TODO
    # 'turbine_array/run_fixed_mesh.py',  # TODO: Currently takes too long to run
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
