import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from adapt_utils.plotting import *


offset = False

di = 'outputs/dwr/enrichment'
fname = os.path.join(di, '{:s}.p'.format('offset' if offset else 'aligned'))
out = pickle.load(open(fname, 'rb'))

out['GE_hp']['label'] = 'GE$_{hp}$'
out['GE_h']['label'] = 'GE$_h$'
out['GE_p']['label'] = 'GE$_p$'
out['DQ']['label'] = 'DQ'

fig, axes = plt.subplots(figsize=(6, 5))
for method in out.keys():
    axes.loglog(out[method]['num_cells'], out[method]['time'], '--x', label=out[method]['label'])
axes.set_xlabel("Element count")
axes.set_ylabel("Computation time [$\mathrm s$]")
axes.legend()
axes.grid(True)
plt.tight_layout()

fig, axes = plt.subplots(figsize=(6, 5))
for method in out.keys():
    if method == 'DQ':
        continue
    # I_eff = np.array(out[method]['effectivity'])/np.array(out[method]['num_cells'])
    I_eff = np.array(out[method]['effectivity'])
    # axes.semilogx(out[method]['num_cells'], I_eff, '--x', label=out[method]['label'])
    axes.loglog(out[method]['num_cells'], I_eff, '--x', label=out[method]['label'])
axes.set_xlabel("Element count")
axes.set_ylabel("Effectivity index")
axes.legend()
axes.grid(True)
plt.tight_layout()

plt.show()
