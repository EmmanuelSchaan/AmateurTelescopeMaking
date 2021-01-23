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
rCenter = 0.5 * (rMin + rMax)
diskRadius = 0.5 * (rMax - rMin)

fullTravel = 2. * (D/2. - rMax)

nDisk = 6

########################################################################


fig=plt.figure(0, figsize=(6,6))
ax=fig.add_subplot(111)
ax.axis('scaled')
#
# Outline of the pitch lap
circle = plt.Circle((0., 0.), 0.5*D, ec='k', fc='none')
ax.add_artist(circle)
#
# sub-diameter disks
for iDisk in range(nDisk):
   circle = plt.Circle((rCenter * np.cos(iDisk * 2.*np.pi/nDisk), rCenter * np.sin(iDisk * 2.*np.pi/nDisk)), 
         diskRadius, ec='k', fc='none')
   ax.add_artist(circle)
#
# outline of the sub-diameter disks
circle = plt.Circle((0., 0.), rMin, ec='k', fc='none', ls='--', lw=1)
ax.add_artist(circle)
circle = plt.Circle((0., 0.), rMax, ec='k', fc='none', ls='--', lw=1)
ax.add_artist(circle)
#
# Outline of the stroke limits
circle = plt.Circle((0., 0.), rMin-fullTravel/2., ec='k', fc='none', ls='--', lw=1)
ax.add_artist(circle)
circle = plt.Circle((0., 0.), rMax+fullTravel/2., ec='k', fc='none', ls='--', lw=1)
ax.add_artist(circle)

#
ax.set_title("Disks")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax.set_xlim((-D/2., D/2.))
ax.set_ylim((-D/2., D/2.))
#
fig.savefig('./figures/pitch_lap_rings/disks_outline_'+testRef+'_ndisk'+str(nDisk)+'.pdf', bbox_inches='tight')
#fig.clf()
plt.show()

