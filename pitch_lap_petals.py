''' This code designs the petal pattern 
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
testRef = "20210109wire"
RPlot = np.array([20.066, 54.864, 79.6925, 98.044, 113.411, 123.7615])  # [mm]
zMirrorVSPara = np.array([-5.62515259e-05,  4.71666935e-05,  4.91025210e-05,  3.07865141e-05,-1.17926334e-05, -5.90311465e-05])  # [mm]
#zMirrorVSCirc = np.array([-5.04072668e-04, -1.47825861e-04,  2.66404781e-05,  1.25618995e-05, -1.87242814e-04, -4.60189337e-04])   # [mm]


# since my polishing tends to preserve my mirror shape,
# I should probably design the petals based on the deviation
# of my mirror to the desired parabola, not the sphere.


# set minimum to zero
zMirrorDeviation = zMirrorVSPara - np.min(zMirrorVSPara) # [mm]

########################################################################
# interpolate and smooth

fzMirrorDeviation = interp1d(RPlot, zMirrorDeviation, kind='quadratic', bounds_error=False, 
                              fill_value=(zMirrorDeviation[0], zMirrorDeviation[-1]))#fill_value='extrapolate')

R = np.linspace(0., D/2., 101)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.axhline(0., lw=1, c='gray')
ax.plot(R, 1.e3 * fzMirrorDeviation(R))
ax.plot(RPlot, 1.e3 * zMirrorDeviation, 'bo')
#
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z$ deviation from desired figure [$\mu$m]')
#
fig.savefig('./figures/pitch_lap_petals/z_deviation_'+testRef+'.pdf', bbox_inches='tight')

plt.show()


########################################################################
# Infer petal pattern edge

nPetal = 5

# max mirror deviation
zMirrorDeviationMax = np.max(fzMirrorDeviation(R))

def polishingEfficiency(r):
   ''' Outputs value in [0., 1.]
   r: mirror zone [mm]
   '''
   result = fzMirrorDeviation(r)
   result /= zMirrorDeviationMax
   return result


def phiPetalEdge(r, eMin=0.05, eMax=1.):
   '''
   r: mirror zone [mm]
   '''
   
   phiMin = eMin * np.pi / nPetal
   phiMax = eMax * np.pi / nPetal

   result = (1. - polishingEfficiency(r)) * (phiMax-phiMin)
   result += phiMin
   return result

'''
fig=plt.figure(0, figsize=(6,6))
ax=fig.add_subplot(111)
#
plt.polar(phiPetalEdge(R), R)
#
ax.set_title("Petal pattern")
ax.set_xlim((-D/2., D/2.))
ax.set_ylim((-D/2., D/2.))
ax.axis('scaled')

plt.show()
'''

########################################################################
# full petal pattern


@np.vectorize
def petalPattern(r, phi, isPitch=True):
   '''
   Returns 1 if the coordinates are part of the requested area (pitch or paper)
   r: mirror zone [mm]
   '''
   if r>D/2.:
      return 0.

   phi %= 2.*np.pi / nPetal

   if phi<=np.pi / nPetal:
      return phi > phiPetalEdge(r)
   else: 
      return phi < 2.*np.pi / nPetal - phiPetalEdge(r)
   



# Cartesian coordinates [mm]
nX = 501
x = np.linspace(-D/2., D/2., nX)
y = np.linspace(-D/2., D/2., nX)

xx, yy = np.meshgrid(x, y, indexing='ij')
rr = np.sqrt(xx**2 + yy**2)
pphi = np.arctan2(yy, xx) + np.pi

# pcolor wants x and y to be edges of cell,
# ie one more element, and offset by half a cell
dX = D / (nX-1)
xEdges = dX * (np.arange(nX+1) - 0.5)
yEdges = xEdges.copy()
xxEdges, yyEdges = np.meshgrid(xEdges, yEdges, indexing='ij')

# compute petal pattern
image = petalPattern(rr, pphi)


fig=plt.figure(0, figsize=(6,6))
ax=fig.add_subplot(111)
#
# outline of the mirror
circle = plt.Circle((0.5*D, 0.5*D), 0.5*D, ec='k', fc='none')
ax.add_artist(circle)
#
# zone obstructed by the secondary mirror
circle = plt.Circle((0.5*D, 0.5*D), 2.6/2.*inchToMm, ec='k', fc='none', ls='--')
ax.add_artist(circle)
#
# Petal pattern
cp=ax.pcolormesh(xxEdges, yyEdges, image, linewidth=0, rasterized=True, vmin=0., vmax=1.)
#cp.set_cmap('binary_r')
cp.set_cmap('YlOrBr')
#
# add legend entries
ax.fill_between([], [], [], facecolor=plt.cm.YlOrBr(1.), label='pitch')
ax.fill_between([], [], [], facecolor=plt.cm.YlOrBr(0.), label='paper')
ax.legend(loc=4, fontsize='xx-small', labelspacing=0., handlelength=1., framealpha=1)
#
ax.set_title("Petal pattern")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax.set_xlim((-D/2., D/2.))
ax.set_ylim((-D/2., D/2.))
ax.axis('scaled')
#
fig.savefig('./figures/pitch_lap_petals/petal_pattern_n'+str(nPetal)+'_'+testRef+'.pdf', bbox_inches='tight')

plt.show()
