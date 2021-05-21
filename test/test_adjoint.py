"""
Runs all discrete and continuous adjoint test case scripts found in `steady/test_cases`,
`unsteady/test_cases/` and `case_studies/`. In some cases these examples include assertions which verify
desired behaviour, but in most cases it is just verified that the solve does not crash and reaches the
end of the required time period.

The code used in this script was largely copied from `thetis/test/examples.py`.
"""
import pytest
import os
import sys
import glob
import subprocess
import shutil


examples = [
    'steady/test_cases/point_discharge2d/run_adjoint.py',
    'steady/test_cases/point_discharge2d/calibrate_radius.py',
    'unsteady/test_cases/idealised_desalination/run_adjoint.py',
    'unsteady/test_cases/solid_body_rotation/run_adjoint.py',
    'case_studies/tohoku/hazard/run_continuous_adjoint.py',
    'case_studies/tohoku/inversion/1d/gradient_continuous.py',
    'case_studies/tohoku/inversion/1d/gradient_discrete.py',
]

cwd = os.path.abspath(os.path.dirname(__file__))
home_dir = os.path.abspath(os.path.join(cwd, '..'))
examples = [os.path.join(home_dir, f) for f in examples]


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
