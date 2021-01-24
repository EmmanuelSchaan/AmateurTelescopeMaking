import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d
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
# Telescope specs

# Blank diameter [mm]
D = 10. * inchToMm

# Focal length [mm]
fNumber = 5.   # "f number"
lf = D*fNumber # [mm]

# Depth of the blank [mm]
depth = D / 6.


########################################################################
# Mirror specifications

'''
print "Primary diameter =", D, "mm =", D / inchToMm, "inch"
print "Aperture area =", np.round(np.pi*(0.1*D/2.)**2, 1), "cm^2,"
print "ie ", np.int((D/7.)**2), "times the area of fully dilated eye pupil (7mm)"
print "ie limiting apparent magnitude is", np.round(6 + 2.512*np.log10((D/7.)**2), 1)
print "compared to 6 for the naked eye."

resBlueArcsec = 1.22 * 400.e-9/(D*1.e-3) * (180.*3600./np.pi)
resRedArcsec = 1.22 * 800.e-9/(D*1.e-3) * (180.*3600./np.pi)
print "Diffraction limited Airy disk radius (Rayleigh criterion):"
print np.round(resBlueArcsec, 2), "arcsec in blue and ", np.round(resRedArcsec, 2), "arcsec in red"
print "Diffraction limited Airy disk diameter:"
print np.round(2.*resBlueArcsec, 2), "arcsec in blue and ", np.round(2.*resRedArcsec, 2), "arcsec in red"

print ""
print "f number =", fNumber
print "focal length =", lf, "mm =", lf / inchToMm, "inch"
print "Min useful magnification, for fully dilated eye pupil (7mm) =", np.round(D/7.,1)
print "ie max useful eyepiece focal length =", np.round(7. * fNumber,1), "mm"
print "Max useful magnification is 30 * (D/1inch) = ", np.round(30. * D/inchToMm, 1)
print "ie min useful eyepiece focal length =", np.round(lf / (30. * D/inchToMm),1), "mm"
print "Magnification =", np.round(lf/5.,1), np.round(lf/10.,1), np.round(lf/20.,1), np.round(lf/35.,1), "for 5 10 20 30mm eyepiece focal length"
'''

########################################################################
# Equations for the circle and parabola, for a given circle radius Rc

def zCircCenter(Rc):
   """Height of the circle center [mm]
   Rc: circle radius [mm]
   """
   return np.sqrt(Rc**2 - (D/2.)**2)


def zCirc(r, Rc):
   """Equation for the circle.
   Rc: circle radius [mm]
   r: polar coordinate [mm]
   Output: height z [mm]
   """
   result = zCircCenter(Rc)
   result -= np.sqrt(Rc**2 - r**2)
   return result

def dzCirc(r, Rc):
   """Slope for the circle.
   Rc: circle radius [mm]
   r: polar coordinate [mm]
   Output: dz/dr [dimless]
   """
   result = 2. * r
   result /= 2. * np.sqrt(Rc**2 - r**2)
   return result


def rContact(Rc):
   """Polar radius [mm] of the contact point between the circle and parabola.
   """
   return np.sqrt(Rc**2 - 4.*lf**2)

def sagittaCirc(Rc):
   """Maximum depth of the sphere [mm],
   given the radius Rc [mm]
   """
   result = Rc - zCircCenter(Rc)
   return result


########################################################################
# Equation for parabola

def zParaOffset(Rc, lf=lf):
   """Vertical offset [mm] for the parabola,
   to be tangent to the circle at the desired polar radius.
   Rc: circle radius [mm]
   """
   rC = rContact(Rc)
   result = zCircCenter(Rc)
   result -= np.sqrt(Rc**2 - rC**2)
   result -= rC**2 / (4.*lf)
   return result

def zFocus(Rc, lf=lf):
   """Height [mm] of the focal point of the parabola,
   given the circle radius Rc [mm]
   """
   result = zParaOffset(Rc) + lf
   return result

def zPara(r, Rc, lf=lf):
   """Equation for the parabola,
   with vertical offset determined to be tangent to the circle of radius Rc
   """
   result = zParaOffset(Rc)
   result += r**2 / (4. * lf)
   return result

def dzPara(r, Rc, lf=lf):
   """First derivative of the parabola
   """
   result = 2. * r / (4. * lf)
   return result

def d2zPara(r, Rc, lf=lf):
   """Second derivative of the parabola
   """
   result = 2. / (4. * lf)
   return result

def rCurv(r, Rc, lf=lf):
   """Radius of curvature of the parabola
   """
   result = (1. + dzPara(r, Rc, lf=lf)**2)**(3./2.)
   result /= np.abs(d2zPara(r, Rc))
   return result


########################################################################
# Find the circle radius Rc, to have the desired rContact = alpha * D/2
# Wisdom says alpha = 0.7

def RcForContact(alpha):
   """Find Rc needed for contact at rContact = alpha * D/2
   """
   result = (alpha*D/2.)**2
   result += 4. * lf**2
   return np.sqrt(result)

# Typical wisdom is to have the contact point at 70%
# of the radius of the blank
Rc70 = RcForContact(0.7)


########################################################################
# Find the circle radius that minimizes the volume between the parabola and the circle

def dVolumedRc(Rc):
   """Quantity proportional to the derivative of the volume between circle and parabola,
   with respect to the circle radius.
   Find the root of this function to minimize the volume between the sphere
   and paraboloid.
   """
   beta = Rc / (D/2.)
   result = 1. - np.sqrt(1. - 1./beta**2)
   result -= 1. / (8.*beta*fNumber)
   return result


# guess for the circle radius: twice the focal length
RcGuess = 2. * D * fNumber
RcBest = optimize.brentq(dVolumedRc , 0.9*RcGuess, 1.1*RcGuess)
rContactBest = rContact(RcBest)
alphaBest = rContactBest/(D/2.)


print ""
print "To minimize the volume between sphere and paraboloid,"
print "the contact point should be at a fraction", np.round(alphaBest, 4), "of the blank radius."
print "The wisdom is 0.7."

print "Hence the best circle radius =", np.round(RcBest, 3), "mm =", np.round(RcBest/inchToMm, 3), "inch"
print "The guess would be twice the focal length, ie", RcGuess, "mm =", np.round(RcGuess/inchToMm, 3), "inch"

print "Circle sagitta =", np.round(sagittaCirc(RcBest),5), "mm =", np.round(sagittaCirc(RcBest) / inchToMm,5), "inch"


########################################################################
# plot the mirror curve
'''
R = np.linspace(-D/2., D/2., 501)
ZCirc = zCirc(R, RcBest)
ZPara = zPara(R, RcBest)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
# Circle
ax.plot(R*0.1, ZCirc, label=r'Circle')
# Parabola
ax.plot(R*0.1, ZPara, label=r'Parabola')
#
# Contact points
ax.plot(np.array([-rContactBest, rContactBest])*0.1, [zCirc(rContactBest, RcBest), zCirc(rContactBest, RcBest)], 'ko')
#
# Blank
ax.plot(R*0.1, np.zeros_like(R), 'k--')
ax.plot(R*0.1, -depth * np.ones_like(R), 'k--')
ax.plot(np.array([-D/2., -D/2.])*0.1, [-depth, 0.], 'k--')
ax.plot(np.array([D/2., D/2.])*0.1, [-depth, 0.], 'k--')
#
ax.legend(loc=3)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z$ [mm]')
#
fig.savefig("./figures/circle.pdf", bbox_inches='tight')



fig=plt.figure(1)
ax=fig.add_subplot(111)
#
# Show tolerances for red, green and blue wavelengths:
# should be wavelength/8 to reach the Rayleigh criterion for a miror,
# and wavelength/4 for a lens
tol = 8.
ax.fill_between(R*0.1, -800.e-3/tol, 800.e-3/tol, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(R*0.1, -600.e-3/tol, 600.e-3/tol, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(R*0.1, -400.e-3/tol, 400.e-3/tol, edgecolor=None, facecolor='b', alpha=0.2)
#
# Compare circle and parabola
ax.plot(R*0.1, 0.*R, 'k--', label=r'Circle')
ax.plot(R*0.1, (ZPara - ZCirc)*1.e3, 'k', label=r'Parabola')
#
# Contact points
ax.plot(np.array([-rContactBest, rContactBest])*0.1, [0., 0.], 'ko')
#
ax.legend(loc=2, fontsize='x-small')
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z_\text{Parabola} - z_\text{Circle}$ [$\mu$m]')
#
fig.savefig("./figures/parabola_vs_circle.pdf", bbox_inches='tight')

plt.show()
'''


########################################################################
########################################################################
# "Spherical aberration" of the parabolic mirror


def zIntercept(zS, r0, RcBest, z0=None, dz0=None):
   '''Compute the height of the tip of the reflected cone, when
   zS: height of the light source (LED) [mm]
   r0: the light reflects on the mirror annulus at radius r0
   RcBest: defines the parabolic mirror curve
   z0 (optional): height of the mirror at r0. default: parabolic mirror
   dz0 (optional): derivative of the mirror at r0. default: parabolic mirror
   '''
   # given the parabolic mirror curve,
   # height of the mirror annulus at r0 [mm]
   if z0 is None:
      z0 = zPara(r0, RcBest)
   if dz0 is None:
      dz0 = dzPara(r0, RcBest)
   # vector u, incident light vector
   # from the light source to the mirror annulus
   u = np.array([r0 - 0., z0 - zS])
   # vector t, tangent to the parabolic mirror
   # at the given annulus, outwards
   t = np.array([1., dz0])
   t /= np.sqrt(np.sum(t**2))
   # vector n, normal to the parabolic mirror
   # at the given annulus, upwards
   n = np.array([-dz0, 1.])
   n /= np.sqrt(np.sum(n**2))
   # vector v, reflected light vector
   # from the mirror annulus to the intercept point
   v = np.dot(u, t) * t - np.dot(u, n) * n
   # height of the intercept,
   # where the reflected light hits the optical axis
   result = z0 - r0 * v[1]/v[0]
   return result

'''
# Zones on the disk
R0 = np.linspace(1.e-5*D/2., D/2., 101)
# consider a light source at the center of the best Sphere
zS = zCircCenter(RcBest)
# computing the heights of the focus points for each zone
f = lambda r0:  zIntercept(zS, r0, RcBest)
ZFocus = np.array(map(f, R0))

# Plot it!
fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.plot(R0, ZFocus, 'b', label=r'Parabola')
ax.axhline(zCircCenter(RcBest), label='Circle')
#
ax.legend(loc=2)
ax.set_xlim((0., D/2.))
ax.set_xlabel(r'Zone radius $r$ [mm]')
ax.set_ylabel(r'Height of focus point [mm]')
ax.set_title(r'Spherical aberration')
#
fig.savefig("./figures/spherical_aberration.pdf", bbox_inches='tight')
#fig.clf()
plt.show()
'''


########################################################################
########################################################################
# Ronchi simulator


lambdaFringe = 0.254 # [mm]

def cosFringe(x, lambdaFringe=lambdaFringe):
   ''' Fringe pattern on the grating
   x: cartesian coordinate [mm]
   lambdaFringe: period of the grating pattern [mm]
   Centered on one dark fringe.
   '''
   return 0.5 * (1. - np.cos(2.*np.pi/lambdaFringe * x))

def stepFringe(x, lambdaFringe=lambdaFringe):
   ''' Fringe pattern on the grating
   x: cartesian coordinate [mm]
   lambdaFringe: period of the grating pattern [mm]
   Centered on one dark fringe.
   '''
   return 1.*((x/lambdaFringe-0.25) % 1 <0.5) #-1.*(x/lambdaFringe % 1 >0.5)


def createImage(zS, zGrating, test='Ronchi', fringe=stepFringe, lambdaFringe=lambdaFringe, nX=2001):
   '''Compute a Ronchi image
   zS: height of light source (LED) [mm]
   zGrating: height of the grating [mm]
   fringe: function defining the fringe pattern on the grating
   nX: number of pixels in image
   '''

   # Cartesian coordinates [mm], not angular coordinates
   # this way, for a perfect sphere, the stripes are perfectly parallel and vertical
   x = np.linspace(-D/2., D/2., nX)
   y = np.linspace(-D/2., D/2., nX)

   ## distortions to fit an actual real life photo
   ## apply a rotation if needed
   #rot = [[cos t, -sint], [sin t, cos t]]
   ## apply a shear if needed
   #shear = [[1. + gamma1, gamma2], [gamma2, 1. - gamma1]]
   ## apply a translation
   #x += dx
   #y += dy

   xx, yy = np.meshgrid(x, y, indexing='ij')
   rr = np.sqrt(xx**2 + yy**2)

   # height of the mirror surface at a given radius
   zz0 = zPara(rr, RcBest)

   f = np.vectorize( lambda r: zIntercept(zS, r, RcBest) ) 
   # for speed, interpolate on 1d array
   R = np.linspace(0., D, 2*nX)
   F = f(R)
   fInterp = interp1d(R, F, kind='linear', bounds_error=False, fill_value=0.)
   # then evaluate on 2d array
   zzIntercept = fInterp(rr)  #np.array(map(f, rr))
   
   if test=='Ronchi':
#      image = fringe(xx * np.abs(1. - zGrating/zzIntercept), lambdaFringe=lambdaFringe)
      image = fringe(xx * np.abs(zzIntercept - zGrating)/(zzIntercept - zz0), lambdaFringe=lambdaFringe)
   elif test=='Foucault':
      image = (zzIntercept >= zGrating) * (xx>=0.) + (zzIntercept <= zGrating) * (xx<0.)
      #image = 2. * image - 1.
   elif test=='Manu':
      image = fringe(xx * np.abs(zzIntercept - zGrating)/(zzIntercept - zz0), lambdaFringe=lambdaFringe)
      image *= fringe(yy * np.abs(zzIntercept - zGrating)/(zzIntercept - zz0), lambdaFringe=lambdaFringe)
   elif test=='Wire':
      width = 0.05   # [mm], determined by (but not equal to) the wire thickness
      image = 1. * (np.abs(zzIntercept-zGrating)>width) * (np.abs(xx)>width)

   image *= (rr < D/2.)

   # pcolor wants x and y to be edges of cell,
   # ie one more element, and offset by half a cell
   dX = D / (nX-1)
   xEdges = dX * (np.arange(nX+1) - 0.5)
   yEdges = xEdges.copy()
   xxEdges, yyEdges = np.meshgrid(xEdges, yEdges, indexing='ij')
   
   # return arrays needed by pcolormesh to plot
   return xxEdges, yyEdges, image


########################################################################
########################################################################
# Ronchi/Foucault: Single frame


'''
# Ronchi test
# height of the Ronchi grating:
# just a bit below the center of the sphere
#zGrating = zCircCenter(RcBest) * 1.01#0.99
zGrating = zCircCenter(RcBest) + 20 # [mm]
# the fixed light source should be at the center of the circle,
# not at the focal point
#!!! When varying zGrating later, zS can also be varied if the LED and grating are attached
#!!! make it possible to change the curve: parabola, sphere, hyperbola
zS = zCircCenter(RcBest)
xxEdges, yyEdges, image = createImage(zS, zGrating, test='Ronchi', fringe=stepFringe, nX=1001)
'''

'''
# Foucault
zGrating = zCircCenter(RcBest)
zS = 1. * zGrating
xxEdges, yyEdges, image = createImage(zCircCenter(RcBest), zCircCenter(RcBest), test='Foucault')
'''

'''
fig=plt.figure(0, figsize=(6,6))
ax=fig.add_subplot(111)
#
# outline of the mirror
circle = plt.Circle((0.5*D, 0.5*D), 0.5*D, ec='w', fc='none')
ax.add_artist(circle)
# mirror test
cp=ax.pcolormesh(xxEdges, yyEdges, image, linewidth=0, rasterized=True)
cp.set_cmap('binary_r')
ax
#
ax.set_title("Ronchi test")
ax.set_xlabel(r"$f/$"+str(fNumber)+r", $D=$"+str(D)+"mm, $l_f=$"+str(round(lf, 1))+"mm"+"\n"+"Grating \& light move together"+" \n"+r"$\lambda_\text{fringe}=$"+str(lambdaFringe)+r"mm, $\delta z_\text{grating}=$"+str(round(zGrating-zCircCenter(RcBest), 1))+"mm", fontsize=16)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax.set_xlim((-D/2., D/2.))
ax.set_ylim((-D/2., D/2.))
ax.axis('scaled')

plt.show()
'''

########################################################################
########################################################################
# Ronchi/Foucault: Animations

"""
# video parameters
fps = 30 # [1/sec]
duration = 3   # [sec]
nFrames = fps * duration   # [dimless]

def CreateMovie(saveFrame, nFrames, fps, test='Ronchi', fixedLight=True, fringe=stepFringe, nX=1001):
   '''Generate all the frames, create the video,
   then delete the individual frames.
   '''
   # file name for the final animation
   if fixedLight:
      name = test + "_fixedlight"
   else:
      name = test + "_samelightgrating" 

   print("Generate all frames")
   f = lambda iFrame: saveFrame(iFrame, './figures/_tmp%05d.jpg'%iFrame, test=test, fixedLight=fixedLight)
   pool = ProcessPool(nodes=3)
   pool.map(f, range(nFrames))

   print("Resize images")
   # resize the images to have even pixel sizes on both dimensions, important for ffmpeg
   # this commands preserves the aspect ratio, rescales the image to fill HD as much as possible,
   # without cropping, then pads the rest with white
   for iFrame in range(nFrames):
      fname = './figures/_tmp%05d.jpg'%iFrame 
      #os.system("convert "+fname+" -resize 1280x720 -gravity center -extent 1280x720 -background white "+fname)
      os.system("convert "+fname+" -resize 1000x1000 -gravity center -extent 1000x1000 -background white "+fname)

   # delete old animation
   os.system("rm ./figures/"+name+".mp4")
   print("Create new animation")
   #os.system("ffmpeg -r "+str(fps)+" -i ./figures/_tmp%05d.jpg -s 1280x720 -vcodec libx264 -pix_fmt yuv420p ./figures/ronchi.mp4")
   os.system("ffmpeg -r "+str(fps)+" -i ./figures/_tmp%05d.jpg -s 1000x1000 -vcodec libx264 -pix_fmt yuv420p ./figures/"+name+".mp4")


   # delete images
   os.system("rm ./figures/_tmp*.jpg")






# Physical param to vary during animation,
# here zGrating [mm]
tMin = -20.
tMax = 20.

# Update function for the plot
def saveFrame(iFrame, fname, test='Ronchi', fixedLight=True, fringe=stepFringe, nX=1001):
   '''Updates the plot as needed for the frame iFrame.
   '''

   # Translate iFrame into the physical param of interest
   x = float(iFrame) / (nFrames - 1)
   t = tMin * (1.-x) + tMax * x
   print "frame="+str(iFrame)+", value="+str(t)

   # Update the image
   zGrating = zCircCenter(RcBest) + t # [mm]
   if fixedLight:
      # Here the grating moves, but the light source is fixed
      zS = zCircCenter(RcBest)
   else:
      # here the grating and the light move together
      zS = zGrating
   xxEdges, yyEdges, image = createImage(zS, zGrating, test=test, fringe=fringe, nX=nX)

   # Create the plot 
   fig=plt.figure(0, figsize=(6,6), dpi=300)
   ax=fig.add_subplot(111)
   #
   # outline of the mirror
   circle = plt.Circle((0.5*D, 0.5*D), 0.5*D, ec='w', fc='none')
   ax.add_artist(circle)
   # mirror test
   cp=ax.pcolormesh(xxEdges, yyEdges, image, linewidth=0, rasterized=True)
   cp.set_cmap('binary_r')
   #
   ax.set_title(test+" test")
   ax.set_xlabel(r"$f/$"+str(fNumber)+r", $D=$"+str(D)+"mm, $l_f=$"+str(round(lf, 1))+"mm"+"\n"+"Grating \& light move together"+" \n"+r"$\lambda_\text{fringe}=$"+str(lambdaFringe)+r"mm, $\delta z_\text{grating}=$"+str(round(zGrating-zCircCenter(RcBest), 1))+"mm", fontsize=16)
   plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
   plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
   ax.set_xlim((-D/2., D/2.))
   ax.set_ylim((-D/2., D/2.))
   ax.axis('scaled')
   #
   fig.savefig(fname, dpi=300)
   fig.clf()
   

# Generate the movies

# Ronchi, fixed light
CreateMovie(saveFrame, nFrames, fps, test='Ronchi', fixedLight=True, fringe=stepFringe, nX=1001)
# Ronchi, light and grating move together
CreateMovie(saveFrame, nFrames, fps, test='Ronchi', fixedLight=False, fringe=stepFringe, nX=1001)

# Foucault, fixed light
CreateMovie(saveFrame, nFrames, fps, test='Foucault', fixedLight=True, fringe=stepFringe, nX=1001)
# Foucault, light and grating move together
CreateMovie(saveFrame, nFrames, fps, test='Foucault', fixedLight=False, fringe=stepFringe, nX=1001)

# Wire test, fixed light
CreateMovie(saveFrame, nFrames, fps, test='Wire', fixedLight=True, fringe=stepFringe, nX=1001)
# Wire test, light and grating move together
CreateMovie(saveFrame, nFrames, fps, test='Wire', fixedLight=False, fringe=stepFringe, nX=1001)

## Manu, fixed light
#CreateMovie(saveFrame, nFrames, fps, test='Manu', fixedLight=True, fringe=stepFringe, nX=1001)
## Manu, light and grating move together
#CreateMovie(saveFrame, nFrames, fps, test='Manu', fixedLight=False, fringe=stepFringe, nX=1001)
"""


########################################################################
########################################################################
# Wire test: infer surface accuracy

########################################################################
# Measured wire test offsets, and uncertainties


# test case, should correspond to circle
#dZWireMeas = - np.array([0. ,0., 0., 0., 0.])   # [inches]
#parTruth = np.zeros(6)

# Expected result for a perfect parabola?
#dZWireMeas = np.linspace(0., 2. * (D/2.)**2/RcBest, 5)
#dZWireMeas = np.linspace(0., 10., 5)


'''
# 2020/03/06
# source and wire move together
testRef = "20200306wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions
dZWireMeas = - np.array([0.04 ,0.015, 0., -0.02, -0.04]) * inchToMm  # convert from [inch] to [mm]
# Uncertainties
sdZWireMeas = 0.005 * np.ones_like(dZWireMeas) * inchToMm   # convert from [inch] to [mm]
'''
'''
# 2020/11/17, wire test
# source is fixed
testRef = "20201117wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([12.09833333, 10.10233333, 8.62583333, 7.17466667, 5.59583333])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.13419223, 0.10556462, 0.04963338, 0.05598115, 0.06667187]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/11/17, foucault test
# source is fixed
testRef = "20201117foucault"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([8.04483333, 5.46683333, 4.1965, 2.78966667, 1.21616667])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.095966, 0.13713426, 0.17351345, 0.12624007, 0.09364902]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/11/21, wire test
# source is fixed
testRef = "20201121wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([21.6015, 19.742, 18.268, 16.96466667, 15.641])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.12519152, 0.08253686, 0.05352258, 0.11355566, 0.09102564]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/11/22, wire test
# source is fixed
testRef = "20201122wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([13.89183333, 11.98733333, 10.60183333, 9.4325, 8.01433333])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.0963698, 0.0791005, 0.09111424, 0.06779319, 0.08384443]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/11/27, wire test
# source is fixed
testRef = "20201127wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([11.7035, 9.76783333, 8.32466667, 6.913, 5.44233333])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.09205569, 0.04915763, 0.06642958, 0.05149434, 0.09472533]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/11/28, wire test
# source is fixed
testRef = "20201128wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([11.863375, 10.094625, 8.64975, 7.353125, 5.991875])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.2313671, 0.20242402, 0.15316311, 0.15893665, 0.12903143]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/01, wire test
# source is fixed
testRef = "20201201wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([19.41525, 17.65475, 16.20325, 14.9955, 13.77225])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.09869999, 0.11106164, 0.0482979, 0.11696688, 0.05200661]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/05, wire test
# source is fixed
testRef = "20201205wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([18.16475, 16.38775, 14.93, 13.3775, 12.0715])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.11272173, 0.01926623, 0.04526588, 0.05136876, 0.06391596]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/12, wire test, axis 1
# source is fixed
testRef = "20201212wireax1"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([3.14145, 1.36845, -0.09155, -1.69105, -2.7273])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.15003562, 0.05648617, 0.04435299, 0.0674662, 0.10878189]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/12, wire test, axis 2
# source is fixed
testRef = "20201212wireax2"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([3.101, 1.18575, -0.1335, -1.5075, -2.64575])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.11287244, 0.08207771, 0.06363716, 0.04630537, 0.12458531]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/12, wire test, combining both axes
# source is fixed
testRef = "20201212wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([3.121225, 1.2771, -0.112525, -1.599275, -2.686525])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.13429255, 0.11536265, 0.05872298, 0.10849234, 0.12385516]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/18, wire test, (axis 1)
# source is fixed
testRef = "20201218wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([13.1155, 11.23775, 9.7795, 8.32375, 7.076])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.045883, 0.03096268, 0.04792442, 0.03310117, 0.11645385]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/20, wire test, (axis 1)
# source is fixed
testRef = "20201220wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([9.937, 8.07225, 6.66825, 5.133, 3.94375])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.06864037, 0.07089208, 0.0605284, 0.05233546, 0.12566896]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/22, wire test, (axis 1)
# source is fixed
testRef = "20201222wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([15.92875, 14.05125, 12.575, 10.95, 9.6425])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.05079062, 0.06168215, 0.04716991, 0.06745369, 0.07798237]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/23, wire test, (axis 1)
# source is fixed
testRef = "20201223wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([14.7675, 12.82375, 11.36, 9.815, 8.4825])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.04322904, 0.06484356, 0.05820223, 0.02061553, 0.11712707]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2020/12/27, wire test, (axis 1)
# source is fixed
testRef = "20201227wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([16.21625, 14.06625, 12.64, 11.09375, 9.8425])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.1836224, 0.04292071, 0.03409545, 0.0469541, 0.14497845]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2021/01/09, wire test, (axis 1)
# source is fixed
testRef = "20210109wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([17.7325, 15.51875, 14.09375, 12.62625, 11.20375])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.06581223, 0.04721957, 0.06435983, 0.03266784, 0.0491649]) # [mm]
sdZWireMeas /= 2.
'''
'''
# 2021/01/17, wire test, (axis 1)
# source is fixed
testRef = "20210117wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.07, 0.15, 0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
#dZWireMeas = - np.array([16.895, 14.3425, 12.8375, 11.24, 10.05])  # [mm]
dZWireMeas = - np.array([16.56, 16.56, 16.56, 14.3425, 12.8375, 11.24, 10.05])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.335, 0.335, 0.335, 0.0375, 0.0225, 0.005, 0.11]) # [mm]
sdZWireMeas /= 2.
'''

# 2021/01/23, wire test, (axis 1)
# source is fixed
testRef = "20210123wire"
# Mirror zones to be measured
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
nRMeas = len(RMeas)
# Measured wire positions (median of 6 runs)
dZWireMeas = - np.array([13.6625, 12.805, 11.065, 9.51875, 8.35375])  # [mm]
dZWireMeas -= dZWireMeas.mean()
# hack: divide by two because fixed source, to go back to moving source case
dZWireMeas /= 2.
# Uncertainties (standard deviation on the mean of 6 runs)
sdZWireMeas = np.array([0.07717675, 0.04153312, 0.04015595, 0.04532866, 0.10472434]) # [mm]
sdZWireMeas /= 2.







########################################################################
# Infer mirror slopes at the measured zones
# In principle, the wire position measures the mirror height and slope.
# However, the mirror height varies from the circle by less than a micron,
# whereas the slope changes the wire position by ~ millimeters.
# As a result, only the mirror slope matters.

def slopesFromWireTest(RMeas, dZWireMeas, sdZWireMeas):
   '''Infer slopes from the wire test.
   The only approximation made here is that for the whole mirror,
   (z_mirror - z_parabola) ~ micron << z_parabola ~ sagitta ~ mm.
   In this case, the wire test measures directly the slope of the mirror.
   '''
   
   def lnL(par):
      '''Likelihood for the wire test
      Params: overall offset of the wire test, and slopes of the mirror zones measured.
      '''
      # read parameters
      zWireOffset = zCircCenter(RcBest) + par[0]  # overall offset of the measurements [mm]
      dZMirror = dzPara(RMeas, RcBest) + par[1:] # slope of the mirror at the spline nodes [mm]
      
      lnl = 0.
      # for each measurement, compare measured and expected z offsets
      # of the source and wire (moving together)
      for iRMeas in range(nRMeas):
         result = zWireOffset + dZWireMeas[iRMeas]
         result -= zIntercept(zWireOffset + dZWireMeas[iRMeas], # source position
                              RMeas[iRMeas],   # mirror zone
                              0.,  # parameter not used here
                              z0=zPara(RMeas[iRMeas], RcBest),   # mirror height. The same for circle/parabola/my mirror to within a micron.
                              dz0=dZMirror[iRMeas]) # mirror height derivative.
         lnl -= 0.5 * result**2 / sdZWireMeas[iRMeas]**2
      return lnl

   # Find the best fit offset and slopes
   x0 = np.zeros(1+nRMeas)
   res = optimize.minimize(lambda par: -lnL(par), x0, method='Nelder-Mead', tol=1e-8)
   #print res.x

   # Extract the best fit parameters
   zOffset = zCircCenter(RcBest) + res.x[0]
   dzMirrorVsPara = res.x[1:]

   return dzMirrorVsPara


# infer slopes from the wire positions
dzMirrorVsPara = slopesFromWireTest(RMeas, dZWireMeas, sdZWireMeas)
# uncertainties from the measurement
dzMirrorVsParaHigh = slopesFromWireTest(RMeas, dZWireMeas+sdZWireMeas, sdZWireMeas)
dzMirrorVsParaLow = slopesFromWireTest(RMeas, dZWireMeas-sdZWireMeas, sdZWireMeas)


# !!!test: pretend we got the correct slopes for the parabola
#dzMirrorVsPara = 0. * dzPara(RMeas, RcBest)


########################################################################
# Approximate inference of the mirror slope from the wire test
# a la Jean Texereau

# expected wire test positions,
# when the source is moving (half of Texereau, for whom source is fixed)
dZWirePara = 0.5 * RMeas**2 / RcBest
dZWirePara -= np.mean(dZWirePara)

# longitudinal aberration at the center of circle
LambdaC = dZWirePara - dZWireMeas
# longitudinal aberration at the focal point
LambdaF = LambdaC / 4.
# transverse aberration at the focal point
lambdaF = LambdaF * RMeas / lf

# slope with respect to parabola
dzMirrorVsParaTexereau = lambdaF / lf


# Compare the expected and measured wire positions
fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.axhline(0.)
ax.plot(RMeas*0.1, dZWirePara, 'k--', label=r'Expected')
ax.plot(RMeas*0.1, dZWireMeas, 'c', label=r'Measured')
ax.fill_between(RMeas*0.1, dZWireMeas - sdZWireMeas,
                           dZWireMeas + sdZWireMeas, edgecolor='', facecolor='c', alpha=0.5)
#
# Allowed region to keep the geometric aberration
# below the diffraction aberration
# Airy disk radius = 1.22 * lambda / D * lf
diffraction = 1.22*800.e-6/D*lf**2/RMeas*4.
ax.fill_between(RMeas*0.1, -diffraction+dZWirePara, diffraction+dZWirePara, edgecolor=None, facecolor='r', alpha=0.2)
diffraction = 1.22*600.e-6/D*lf**2/RMeas*4.
ax.fill_between(RMeas*0.1, -diffraction+dZWirePara, diffraction+dZWirePara, edgecolor=None, facecolor='g', alpha=0.2)
diffraction = 1.22*400.e-6/D*lf**2/RMeas*4.
ax.fill_between(RMeas*0.1, -diffraction+dZWirePara, diffraction+dZWirePara, edgecolor=None, facecolor='b', alpha=0.2, label=r'Diffraction')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z_\text{wire}$ [mm]')
ax.set_title(r'Wire test positions'+ '\n'+r' (comoving source \& wire)')
#
fig.savefig("./figures/test_positions_"+testRef+".pdf", bbox_inches='tight')

plt.show()


# Longitudinal aberration at focal point
fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.axhline(0.)
ax.plot(RMeas*0.1, LambdaF*1.e3, 'b', label=r'Geometric (measured)')
#
# Compare with aberration from diffraction
# Airy disk radius = 1.22 * lambda / D * lf
ax.fill_between(RMeas*0.1, -1.22*800.e-3/D*lf**2/RMeas, 1.22*800.e-3/D*lf**2/RMeas, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(RMeas*0.1, -1.22*600.e-3/D*lf**2/RMeas, 1.22*600.e-3/D*lf**2/RMeas, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(RMeas*0.1, -1.22*400.e-3/D*lf**2/RMeas, 1.22*400.e-3/D*lf**2/RMeas, edgecolor=None, facecolor='b', alpha=0.2, label=r'Diffraction')
#
ax.legend(loc=1, fontsize='x-small', labelspacing=0.)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'Longitudinal aberration'+'\n'+r'at focal point [$\mu$m]')
#
fig.savefig("./figures/test_longitudinal_aberration_"+testRef+".pdf", bbox_inches='tight')

plt.show()


# Transverse aberration at focal point
# Danjon & Couder criterion 1:
# the geometric transverse aberration should be less than the diffraction
fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.axhline(0.)
#
# Transverse aberration: geometric
result = lambdaF - 0.5*(np.min(lambdaF) + np.max(lambdaF))
result *= 1.e3 # convert to microns
ax.plot(RMeas*0.1, result, 'b', label=r'Geometric (measured)')
#
# Compare with aberration from diffraction
# Airy disk radius = 1.22 * lambda / D * lf
ax.fill_between(RMeas*0.1, -1.22*800.e-3/D*lf, 1.22*800.e-3/D*lf, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(RMeas*0.1, -1.22*600.e-3/D*lf, 1.22*600.e-3/D*lf, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(RMeas*0.1, -1.22*400.e-3/D*lf, 1.22*400.e-3/D*lf, edgecolor=None, facecolor='b', alpha=0.2, label=r'Diffraction')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'Transverse aberration'+'\n'+r'at focal point [$\mu$m]')
ax.set_title(r'Danjon \& Couder criterion 1' + '\n' + r'Geometric VS diffraction spot')
#
fig.savefig("./figures/test_transverse_aberration_"+testRef+".pdf", bbox_inches='tight')

plt.show()





########################################################################
# Compute surface deviation from parabola


def integrateSlopeDeviation(dzMirrorVsPara):
   # Zones at which the mirror heights will be computed
   RPlot = np.concatenate(([0.], RMeas, [D/2.]))
   RPlot = 0.5*(RPlot[:-1] + RPlot[1:])

   # integrate the mirror slopes to get mirror curve
   zMirrorVSPara = np.zeros_like(RPlot)
   for i in range(len(RMeas)):
      zMirrorVSPara[i+1] = zMirrorVSPara[i]
      zMirrorVSPara[i+1] += dzMirrorVsPara[i] * (RPlot[i+1] - RPlot[i])

   # find the offset of the curve that makes it closest to parabola
   def offsetLoss(offset):
      result = zMirrorVSPara.copy() + offset
   #   result -= zPara(RPlot, RcBest) - zCirc(RPlot, RcBest)
      result = np.sum(result**2)
      return result
   res = optimize.minimize(offsetLoss, 0., method='Nelder-Mead', tol=1e-8)
   offset = res.x[0]

   zMirrorVSPara += offset
   return RPlot, zMirrorVSPara


RPlot, zMirrorVSPara = integrateSlopeDeviation(dzMirrorVsPara)
RPlot, zMirrorVSParaTexereau = integrateSlopeDeviation(dzMirrorVsParaTexereau)

# uncertainties from measurement of wire positions
RPlot, zMirrorVSParaHigh = integrateSlopeDeviation(dzMirrorVsParaHigh)
RPlot, zMirrorVSParaLow = integrateSlopeDeviation(dzMirrorVsParaLow)
sZMirrorVSPara = zMirrorVSParaHigh - zMirrorVSParaLow


########################################################################
# Plot measured surface, compared with circle and parabola

fig=plt.figure(0)
ax=fig.add_subplot(111)
#
# Show tolerances for red, green and blue wavelengths:
# should be wavelength/8 to reach the Rayleigh criterion for a miror,
# and wavelength/4 for a lens
tol = 8.
ax.fill_between(RPlot*0.1, -800.e-3/tol, 800.e-3/tol, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(RPlot*0.1, -600.e-3/tol, 600.e-3/tol, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(RPlot*0.1, -400.e-3/tol, 400.e-3/tol, edgecolor=None, facecolor='b', alpha=0.2, label=r'$\pm1/8$-wave')
#
# Compare circle, parabola and measured profile
ax.plot(RPlot*0.1, 1.e3 * (zCirc(RPlot, RcBest) - zPara(RPlot, RcBest)), 'k--', label=r'Circle')
ax.plot(RPlot*0.1, 0. * RPlot, 'k-', label=r'Parabola')
ax.plot(RPlot*0.1, 1.e3 * zMirrorVSPara, 'c', label=r'measured')
ax.fill_between(RPlot*0.1, 1.e3 * zMirrorVSParaLow, 1.e3 * zMirrorVSParaHigh, edgecolor='', facecolor='c', alpha=0.5)
ax.plot(RPlot*0.1, 1.e3 * zMirrorVSParaTexereau, 'b--', label=r'measured Texereau')
#
ax.plot(RMeas*0.1, 0.*RMeas, 'go')
#
ax.legend(loc='upper center', fontsize='x-small', labelspacing=0)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z - z_\text{Parabola}$ [$\mu$m]')
ax.set_title(r'Wire test analysis '+testRef)
#
fig.savefig("./figures/test_mirror_vs_parabola_"+testRef+".pdf", bbox_inches='tight')


fig=plt.figure(1)
ax=fig.add_subplot(111)
#
# Show tolerances for red, green and blue wavelengths:
# should be wavelength/8 to reach the Rayleigh criterion for a miror,
# and wavelength/4 for a lens
tol = 8.
ax.fill_between(RPlot*0.1, -800.e-3/tol, 800.e-3/tol, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(RPlot*0.1, -600.e-3/tol, 600.e-3/tol, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(RPlot*0.1, -400.e-3/tol, 400.e-3/tol, edgecolor=None, facecolor='b', alpha=0.2, label=r'$\pm1/8$-wave')
#
# Compare circle, parabola and measured profile
ax.plot(RPlot*0.1, 0. * RPlot, 'k--', label=r'Circle')
ax.plot(RPlot*0.1, 1.e3 * (zPara(RPlot, RcBest) - zCirc(RPlot, RcBest)), 'k-', label=r'Parabola')
#
zMirrorVSCirc = zMirrorVSPara + zPara(RPlot, RcBest) - zCirc(RPlot, RcBest)
ax.plot(RPlot*0.1, 1.e3 * zMirrorVSCirc, 'c', label=r'measured')
ax.fill_between(RPlot*0.1, 1.e3 * (zMirrorVSParaLow + zPara(RPlot, RcBest) - zCirc(RPlot, RcBest)), 
                           1.e3 * (zMirrorVSParaHigh + zPara(RPlot, RcBest) - zCirc(RPlot, RcBest)), edgecolor='', facecolor='c', alpha=0.5)
ax.plot(RPlot*0.1, 1.e3 * (zMirrorVSParaTexereau + zPara(RPlot, RcBest) - zCirc(RPlot, RcBest)), 'b--', label=r'measured Texereau')
#
ax.plot(RMeas*0.1, 0.*RMeas, 'go')
#
ax.legend(loc='lower center', fontsize='x-small', labelspacing=0)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z - z_\text{Circle}$ [$\mu$m]')
ax.set_title(r'Wire test analysis '+testRef)
#
fig.savefig("./figures/test_mirror_vs_circle_"+testRef+".pdf", bbox_inches='tight')


plt.show()

########################################################################
########################################################################
# Find best-fitting parabola,
# and show the difference


def zMirrorVsParaNew(zMirrorVsPara, RcNew):
   '''Take the measured mirror excess slopes, relative to the target parabola,
   and convert them int excess slopes relative to the new parabola
   '''
   lfNew = 0.5 * RcNew
   zParaNew = zPara(RPlot, RcNew, lfNew)  # slopes of new parabola
   zParaNew -= zPara(RPlot, RcBest)   # deviation from target parabola
   result = zMirrorVsPara - zParaNew
   result -= np.mean(result)  # subtract a constant offset
   return result



def bestFitParabola(zMirrorVsPara):
   '''Given the measured mirror slopes,
   relative to the target parabola,
   find the best-fit parabola radius RcParaBest
   and the slope differences dzMirrorVsParaBest
   with that best fit parabola
   '''

   def loss(x):
      '''x = RcNew
      returns the summed squared difference in slopes
      between the measurements and the new parabola
      '''
      RcNew = RcBest + x[0]
      result = zMirrorVsParaNew(zMirrorVsPara, RcNew)
      #result = np.sum(result**2)
      result = np.max(np.abs(result))
      return result
      
   # find the best fit parabola RcBest
   x0 = [0]
   res = optimize.minimize(lambda x: loss(x), x0, method='Nelder-Mead', tol=1e-8)

   RcParaBest = RcBest + res.x[0]
   zMirrorVsParaBest = zMirrorVsParaNew(zMirrorVsPara, RcParaBest) 

   return RcParaBest, zMirrorVsParaBest


# get the best fit parabola,
# and the corresponding surface deviations
RcParaBest, zMirrorVSParaBest = bestFitParabola(zMirrorVSPara)

# uncertainty from measurement of the wire position
RcParaBestHigh, zMirrorVSParaBestHigh = bestFitParabola(zMirrorVSParaHigh)
RcParaBestLow, zMirrorVSParaBestLow = bestFitParabola(zMirrorVSParaLow)


########################################################################


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
# Show tolerances for red, green and blue wavelengths:
# should be wavelength/8 to reach the Rayleigh criterion for a miror,
# and wavelength/4 for a lens
tol = 8.
ax.fill_between(RPlot*0.1, -800.e-3/tol, 800.e-3/tol, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(RPlot*0.1, -600.e-3/tol, 600.e-3/tol, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(RPlot*0.1, -400.e-3/tol, 400.e-3/tol, edgecolor=None, facecolor='b', alpha=0.2, label=r'$\pm1/8$-wave')
#
# Compare circle, parabola and measured profile
ax.plot(RPlot*0.1, 0. * RPlot, 'k-', label=r'Parabola')
#
ax.plot(RPlot*0.1, 1.e3 * zMirrorVSParaBest, 'c', label=r'measured vs closest')
ax.fill_between(RPlot*0.1, 1.e3 * zMirrorVSParaBestLow,
                           1.e3 * zMirrorVSParaBestHigh, edgecolor='', facecolor='c', alpha=0.5)
#
ax.plot(RPlot*0.1, 1.e3 * zMirrorVSPara, 'b--', label=r'measured vs target')
ax.fill_between(RPlot*0.1, 1.e3 * zMirrorVSParaLow,
                           1.e3 * zMirrorVSParaHigh, edgecolor='', facecolor='b', alpha=0.5)
#
ax.plot(RMeas*0.1, 0.*RMeas, 'go')
#
ax.legend(loc='upper center', fontsize='x-small', labelspacing=0)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z - z_\text{Parabola}$ [$\mu$m]')
ax.set_title(r'Wire test analysis '+testRef)
#
fig.savefig("./figures/test_mirror_vs_closest_parabola_"+testRef+".pdf", bbox_inches='tight')

plt.show()
