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





#########################################################################
## Infer petal pattern edge
#
#
#########################################################################
## full petal pattern
#
#
#
#
## Cartesian coordinates [mm]
#nX = 501
#x = np.linspace(-D/2., D/2., nX)
#y = np.linspace(-D/2., D/2., nX)
#
#xx, yy = np.meshgrid(x, y, indexing='ij')
#rr = np.sqrt(xx**2 + yy**2)
#pphi = np.arctan2(yy, xx) 
##pphi += np.pi
#
## pcolor wants x and y to be edges of cell,
## ie one more element, and offset by half a cell
#dX = D / (nX-1)
#xEdges = dX * (np.arange(nX+1) - 0.5)
#yEdges = xEdges.copy()
#xxEdges, yyEdges = np.meshgrid(xEdges, yEdges, indexing='ij')
#
## compute petal pattern
#image = petalPattern(rr, pphi)
#
#
#fig=plt.figure(0, figsize=(6,6))
#ax=fig.add_subplot(111)
##
## outline of the mirror
#circle = plt.Circle((0.5*D, 0.5*D), 0.5*D, ec='k', fc='none')
#ax.add_artist(circle)
##
## zone obstructed by the secondary mirror
#circle = plt.Circle((0.5*D, 0.5*D), 2.6/2.*inchToMm, ec='k', fc='none', ls='--')
#ax.add_artist(circle)
##
## Petal pattern
#cp=ax.pcolormesh(xxEdges, yyEdges, image, linewidth=0, rasterized=True, vmin=0., vmax=1.)
##cp.set_cmap('binary_r')
#cp.set_cmap('YlOrBr')
##
## add legend entries
#ax.fill_between([], [], [], facecolor=plt.cm.YlOrBr(1.), label='pitch')
#ax.fill_between([], [], [], facecolor=plt.cm.YlOrBr(0.), label='paper')
#ax.legend(loc=4, fontsize='xx-small', labelspacing=0., handlelength=1., framealpha=1)
##
#ax.set_title("Petal pattern")
#plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
#ax.set_xlim((-D/2., D/2.))
#ax.set_ylim((-D/2., D/2.))
#ax.axis('scaled')
##
#fig.savefig('./figures/pitch_lap_petals/petal_pattern_n'+str(nPetal)+'_'+testRef+'.pdf', bbox_inches='tight')
#
#plt.show()
#
