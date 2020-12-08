import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate, stats
#from scipy.interpolate import UnivariateSpline
#import emcee
#import corner

# to delete files, for animations
import os
# for generating animations in parallel
from pathos.multiprocessing import ProcessPool



# unit conversion
inchToMm = 2.54 * 10.


########################################################################

# Mirror radius [mm]
R = 5. * inchToMm

# Weight of top disk [N]
weightTopDisk = 1.


########################################################################
# Geometry: boundary of overlap region between the two disks

def yBoundaryOverlap(x, s):
   '''y coordinate of boundary of the overlap zone
   between the two disks,
   given that their centers are shifted by s [mm],
   with s in [0, R]
   '''
   if s-R<=x and x<s/2.:
      return  np.sqrt(R**2 - (x - s)**2)
   elif s/2.<=x and x<=R:
      return np.sqrt(R**2 - x**2)
   else:
      return 0.


def plotDiskConfig(s):
   fig=plt.figure(0)
   ax=fig.add_subplot(111)
   #
   # bottom disk
   disk1 = plt.Circle((0., 0.), R, fc='r', ec='none', alpha=0.5)
   ax.add_artist(disk1)
   #
   # top disk
   disk2 = plt.Circle((s, 0.), R, fc='b', ec='none', alpha=0.5)
   ax.add_artist(disk2)
   #
   # test boundary
   x = np.linspace(s-R, R, 101)
   f = lambda x: yBoundaryOverlap(x, s)
   y = np.array(map(f, x))
   plt.plot(x, y, 'k--')
   plt.plot(x, -y, 'k--')

   plt.axis('scaled')
   ax.set_xlim((-2.*R, 2.*R))
   ax.set_ylim((-2.*R, 2.*R))

   plt.show()


#plotDiskConfig(R/4.)


########################################################################
# pressure field on the overlap region

def pressureField(x, y, s):
   '''Affine pressure field,
   at position x,y [mm],
   given the shift s [mm],
   and the affine parameters a and b
   '''
   # solve for the affine coefficients,
   # assuming balance of forces and moments
   A = ACoeff(s)
   B = BCoeff(s)
   C = CCoeff(s)
   D = DCoeff(s)
   #
   a = D * weightTopDisk / (A * D - B * C)
   b = -C * weightTopDisk / (A * D - B * C)

   # affine pressure field
   result = a * x + b

   # one can add or subtract a constant pressure,
   # since a uniform pressure wears the mirror evenly
   # and is thus unobservable.
   # This is not quite true if part of the mirror is lifted...
   # and regardless this is no longer the true pressure
   result -= a * (s-R) + b

   # enforce zero pressure outside the overlap region
   result *= (s-R<=x) * (x<=R)
   result *= np.abs(y)<=yBoundaryOverlap(x, s)

   return result



def ACoeff(s):
   '''
   '''
   def integrand(x): return 2. * yBoundaryOverlap(x, s) * x
   return integrate.quad(integrand, s-R, R, epsabs=0., epsrel=1.e-2)[0]

def BCoeff(s):
   '''
   '''
   def integrand(x): return 2. * yBoundaryOverlap(x, s)
   return integrate.quad(integrand, s-R, R, epsabs=0., epsrel=1.e-2)[0]

def CCoeff(s):
   '''
   '''
   def integrand(x): return 2. * yBoundaryOverlap(x, s) * x * (x - s)
   return integrate.quad(integrand, s-R, R, epsabs=0., epsrel=1.e-2)[0]

def DCoeff(s):
   '''
   '''
   def integrand(x): return 2. * yBoundaryOverlap(x, s) * (x - s)
   return integrate.quad(integrand, s-R, R, epsabs=0., epsrel=1.e-2)[0]


def plotPressureField(s):
   '''Visualize the pressure field in 2d,
   and radially averaged for the two disks
   '''


   # 2d plot of pressure field
   fig=plt.figure(0)
   ax=fig.add_subplot(111)
   #
   # bottom disk
   disk1 = plt.Circle((0., 0.), R, fc='none', ec='r', lw=2, alpha=0.5)
   ax.add_artist(disk1)
   #
   # top disk
   disk2 = plt.Circle((s, 0.), R, fc='none', ec='b', lw=2, alpha=0.5)
   ax.add_artist(disk2)
   #
   # pressure field
   # pcolor wants x and y to be edges of cell,
   # ie one more element, and offset by half a cell
   n = 101
   # edges
   xEdges = np.linspace(-2.*R, 2.*R, n)
   yEdges = xEdges.copy()
   xEdges,yEdges = np.meshgrid(xEdges, yEdges, indexing='ij')
   # centers
   dx = 4.*R / (n-1)
   xCen = np.linspace(-2.*R + 0.5*dx, 2.*R - 0.5*dx, n-1)
   yCen = xCen.copy()
   xCen,yCen = np.meshgrid(xCen, yCen, indexing='ij')
   # compute pressure field
   def f(x, y): return pressureField(x, y, s)
   g = np.vectorize(f)
   p = g(xCen, yCen)
   p = np.ma.masked_where(p==0, p)
   #
   cp=ax.pcolormesh(xEdges, yEdges, p, linewidth=0, rasterized=True)
   cp.set_cmap(plt.cm.RdGy_r)
   vMax = np.max(p)
   vMin = np.min(p)
   vAbsMax = max(np.abs(vMin), np.abs(vMax))
   cp.set_clim(-vAbsMax, vAbsMax)
   fig.colorbar(cp)
   #
   plt.axis('scaled')
   ax.set_xlim((-2.*R, 2.*R))
   ax.set_ylim((-2.*R, 2.*R))

   #plt.show()

   # 1d radial averages for the two disks
   fig=plt.figure(1)
   ax=fig.add_subplot(111)
   #



   # define radial bin values
   nBins = 20
   binEdges = np.linspace(0., R, nBins+1)
   # radius wrt disk 1 and 2
   r1 = np.sqrt(xCen**2 + yCen**2)
   r2 = np.sqrt((xCen-s)**2 + yCen**2)
   #
   # mean pressure for disk 1
   binCenters1, binEdges1, binIndices = stats.binned_statistic(r1.flatten(), r1.flatten(), statistic='mean', bins=binEdges)
   p1, binEdges1, binIndices = stats.binned_statistic(r1.flatten(), p.flatten(), statistic='mean', bins=binEdges)
   minP1 = np.min(p1[np.where(np.isfinite(p1))])
   maxP1 = np.max(p1[np.where(np.isfinite(p1))])
   p1 = (p1-minP1) / (maxP1-minP1)
   #
   # mean pressure for disk 2
   binCenters2, binEdges2, binIndices = stats.binned_statistic(r2.flatten(), r2.flatten(), statistic='mean', bins=binEdges)
   p2, binEdges2, binIndices = stats.binned_statistic(r2.flatten(), p.flatten(), statistic='mean', bins=binEdges)
   minP2 = np.min(p2[np.where(np.isfinite(p2))])
   maxP2 = np.max(p2[np.where(np.isfinite(p2))])
   p2 = (p2-minP2) / (maxP2-minP2)

   print p1
   print p2
   
   ax.plot(binCenters1, p1, 'r')
   ax.plot(binCenters2, p2, 'b')

   plt.show()


plotPressureField(R/2.)


