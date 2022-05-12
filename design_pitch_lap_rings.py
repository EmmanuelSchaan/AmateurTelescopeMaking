''' This code draws ring patterns
to be pressed on the pitch lap
to achieve a given polishing profile.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d

# unit conversion
inchToMm = 2.54 * 10.

########################################################################
# Telescope specs

# Blank diameter [mm]
D = 10. * inchToMm

########################################################################
# Mirror profile inferred from the wire test

# 20210109wire
testRef = "20210117wire"
rMin = 40.  # [mm]
rMax = 105. # [mm]


########################################################################


fig=plt.figure(0, figsize=(6,6))
ax=plt.subplot(111, projection='polar')
#
# radial grid
#radii = D/2. * np.linspace(0., 1., 11)
#lines, labels = plt.rgrids(radii, labels=None, angle=22.5, fmt=None)
#
# ring pattern
radii = [rMin, rMax]
lines, labels = plt.rgrids(radii)
#
ax.set_title("Rings")
ax.set_rmax(D/2.)
#ax.grid(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
#
fig.savefig('./figures/pitch_lap_rings/ring_outline_'+testRef+'.pdf', bbox_inches='tight')
#fig.clf()
plt.show()


