from thetis import *

import os
import matplotlib.pyplot as plt

from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


fig, axes = plt.subplots(figsize=(12, 6))
triplot(TurbineArrayOptions().default_mesh, axes=axes, interior_kw={'linewidth': 0.2})
axes.legend()
axes.axis(False)
plt.tight_layout()
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh", ext])))
